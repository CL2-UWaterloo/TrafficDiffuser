# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import copy

from typing import Dict, Mapping

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Bernoulli

from torch_cluster import radius, radius_graph
from torch_scatter import scatter_logsumexp
from torch_geometric.data import Batch, HeteroData


from layers import TransformerDecoderLayerDiff
from layers import FourierEmbedding
from utils import denormalize, weight_init


class HLDiffusion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.guid_task = args.guid_task
        
        self.net = HLDenoiser(
            dataset=args.dataset,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            output_head=args.output_head,
            init_timestep=args.init_timestep,
            num_freq_bands=args.num_freq_bands,
            num_layers=args.num_denoiser_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
            m_dim=args.m_dim,
        )
        self.var_sched = VarianceSchedule(
            num_steps=args.num_diffusion_steps,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            mode='linear',
        )
        probs = torch.tensor([0.5])
        self.B_dist = Bernoulli(probs=probs)

    @staticmethod
    def _attention_entropy(attention_weights, polygon_counts, device):
        """Compute the existing map-attention entropy regularizer."""
        batch_size, num_heads, num_agents, num_map_polygons = attention_weights[0].shape
        padding_mask = torch.ones(
            batch_size, num_map_polygons, device=device, dtype=torch.bool)
        for scene_index, polygon_count in enumerate(polygon_counts):
            padding_mask[scene_index, :polygon_count] = False
        valid_map_mask = (~padding_mask).view(
            batch_size, 1, 1, num_map_polygons).float()

        layer_entropies = []
        denominator = polygon_counts[:, None, None]
        for weights in attention_weights:
            weights = weights * valid_map_mask
            normalized = weights.abs() / (
                torch.norm(weights, dim=-1, keepdim=True) + 1e-6)
            entropy = -torch.sum(
                normalized * torch.log(normalized + 1e-6), dim=-1)
            layer_entropies.append((entropy / denominator).float())
        return torch.stack(layer_entropies).mean(dim=(1, 2, 3)).sum()

    def get_loss(self,
                 clean_state,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 eval_mask=None,
                 ) -> Dict[str, torch.Tensor]:
        device = clean_state.device
        agent_batch = data['agent']['batch'][eval_mask]
        num_scenes = agent_batch[-1].item() + 1
        timesteps = torch.as_tensor(
            self.var_sched.uniform_sample_t(num_scenes), device=device)

        alpha_bar = self.var_sched.alpha_bars[timesteps][:, None]
        beta = self.var_sched.betas[timesteps][:, None][agent_batch]
        signal_scale = torch.sqrt(alpha_bar)
        noise_scale = torch.sqrt(1 - alpha_bar)
        noise = torch.randn_like(clean_state)

        noisy_state = (
            signal_scale[agent_batch] * clean_state
            + noise_scale[agent_batch] * noise)
        predicted_noise, attention_weights = self.net(
            copy.deepcopy(noisy_state), beta, data, scene_enc,
            eval_mask=eval_mask, mode=self.B_dist.sample())

        polygon_counts = data['map_polygon']['batch'].bincount(
            minlength=data.num_graphs)
        entropy = self._attention_entropy(
            attention_weights, polygon_counts, device)
        reconstructed_state = (
            noisy_state - noise_scale[agent_batch] * predicted_noise
        ) / signal_scale[agent_batch]
        diffusion_loss = (noise - predicted_noise) ** 2
        return diffusion_loss, reconstructed_state, entropy

    def sample(self,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               grad_guid = None,
               guid_param = None,
               ) -> Dict[str, torch.Tensor]:
        common_args = (
            data, scene_enc, if_output_diffusion_process, start_data,
            reverse_steps, eval_mask, sampling, stride,
        )
        if self.guid_task == 'none':
            return self.sample_vd(*common_args)
        if grad_guid is None or guid_param is None:
            raise ValueError('Guided sampling requires grad_guid and guid_param')
        return self.sample_guide(
            *common_args, grad_guid=grad_guid, guid_param=guid_param)

    def _decode_endpoint_state(self, traj, data, mean, std):
        """Convert normalized endpoint state to world positions and speeds."""
        eval_mask = data['agent']['mask']
        agent_batch = data['agent']['batch'][eval_mask]
        map_min = data['map_min'].view(-1, 3)[..., :2][agent_batch]
        map_max = data['map_max'].view(-1, 3)[..., :2][agent_batch]

        init_pos, init_speed = denormalize(traj[..., :3], mean, std, map_min, map_max)
        final_pos, final_speed = denormalize(traj[..., 3:], mean, std, map_min, map_max)
        
        return init_pos, final_pos, init_speed, final_speed

    def task_diff(self, task, traj, label, mean, std):            
        if 'map' in task:
            data = label
            center_map = data['map_point']['side'] == 2
            map_point_pos = data['map_point']['position'] [center_map]
                        
            eval_mask = data['agent']['mask']

            pos_init = traj[..., :2]
            pos_final = traj[..., 3:5]
            pos_init_orig_scale, pos_final_orig_scale, _, _ = (
                self._decode_endpoint_state(traj, data, mean, std))
            traj_init_final = torch.cat([pos_init_orig_scale, pos_final_orig_scale], dim=-1).view(-1, 2)
            edge_index_a2m = radius(
                x=map_point_pos[:, :2],
                y=traj_init_final,
                r=torch.inf,
                batch_x=data['map_point']['batch'][center_map] if isinstance(data, Batch) else None,
                batch_y=data['agent']['batch'][eval_mask].unsqueeze(1).repeat(1, 2).view(-1) if isinstance(data, Batch) else None,
                max_num_neighbors=len(data['map_point']['batch']))
            rel_pos_a2m = traj_init_final[edge_index_a2m[0]] - map_point_pos[edge_index_a2m[1], :2]
            k = 10.0
            dist = torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1)
            scaled_dist = - k * dist
            
            log_sum_exp = scatter_logsumexp(scaled_dist, edge_index_a2m[0], dim=0)
            min_dist = -1.0 / k * log_sum_exp
            
            edge_index_a2a = radius_graph(
                x=pos_init_orig_scale,
                r=torch.inf,
                batch=data['agent']['batch'][eval_mask] if isinstance(data, Batch) else None,
                max_num_neighbors=len(data['agent']['batch'][eval_mask]))
            rel_pos_a2m = pos_init[edge_index_a2a[0]] - pos_init[edge_index_a2a[1]]
            dist = torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1)

            dist_init = torch.nn.functional.relu(2 - dist)
            
            
            edge_index_a2a = radius_graph(
                x=pos_final_orig_scale,
                r=torch.inf,
                batch=data['agent']['batch'][eval_mask] if isinstance(data, Batch) else None,
                max_num_neighbors=len(data['agent']['batch'][eval_mask]))
            rel_pos_a2m = pos_final[edge_index_a2a[0]] - pos_final[edge_index_a2a[1]]
            dist = torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1)

            dist_final = torch.nn.functional.relu(2 - dist)
            
            min_dist = nn.ReLU()(min_dist - 2).mean()
            goal_diff = torch.stack([min_dist, dist_init.mean(), dist_final.mean()], dim=-1)
            return goal_diff
        
        if 'original' in task:
            data = label

            eval_mask = data['agent']['mask']
            
            pos_init_orig_scale, pos_final_orig_scale, pred_speed_init, pred_speed_final = (
                self._decode_endpoint_state(traj, data, mean, std))
            
            
            gt_init_pos = data['agent']['scaled_position'][eval_mask, self.net.init_timestep, :2]
            init_speed = torch.norm(data['agent']['velocity'][eval_mask, self.net.init_timestep, :], p=2, dim=-1)
            
            gt_final_pos = data['agent']['scaled_position'][eval_mask, -1, :2]
            final_speed = torch.norm(data['agent']['velocity'][eval_mask, -1, :], p=2, dim=-1)
            
            
            init_dist = torch.norm(pos_init_orig_scale - gt_init_pos, p=2, dim=-1)
            final_dist = torch.norm(pos_final_orig_scale - gt_final_pos, p=2, dim=-1)
        
            init_speed_diff = torch.norm(pred_speed_init - init_speed, p=2, dim=-1)
            final_speed_diff = torch.norm(pred_speed_final - final_speed, p=2, dim=-1)

            goal_diff = torch.stack([init_dist.mean(), final_dist.mean(), init_speed_diff, final_speed_diff], dim=-1)
            return goal_diff

        raise ValueError(f'Unknown guidance task: {task}')
        
    def sample_vd(self, 
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20
               ) -> Dict[str, torch.Tensor]:
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps
        
        device = scene_enc['x_pt'].device

        num_agents = eval_mask.sum()
        
        e_init_rand = torch.randn([num_agents, self.net.m_dim]).to(device)

        if start_data is None:
            x_init_T = e_init_rand
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)
            x_init_T = c0 * start_data.unsqueeze(1) + c1 * e_init_rand
            
        x_init_t_list = [x_init_T]
        torch.cuda.empty_cache()
        wgt_store = []
        for t in range(reverse_steps, 0, -stride):
            z_init = torch.randn_like(x_init_T) if t > 1 else torch.zeros_like(x_init_T)

            beta = self.var_sched.betas[t]
            
            alpha = self.var_sched.alphas[t]    
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            c0 = 1 / torch.sqrt(alpha)
            c1 = (1-alpha) / torch.sqrt(1 - alpha_bar)
            sigma = self.var_sched.get_sigmas(t, 0)
            
            x_init_t = x_init_t_list[-1]
            
            with torch.no_grad():
                beta = beta.unsqueeze(-1).repeat(num_agents, 1).to(device)
                g_init_theta, wgts = self.net(copy.deepcopy(x_init_t), beta, data, scene_enc, eval_mask=eval_mask, mode=0)     
                if t in [10, 100, 200, 300, 400]:
                    wgt_store.append(wgts)

            if sampling == 'ddpm':
                x_init_next = c0 * (x_init_t - c1 * g_init_theta) + sigma * z_init
            elif sampling == 'ddim':
                x0_init_t = (x_init_t - g_init_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_init_next = alpha_bar_next.sqrt() * x0_init_t + (1 - alpha_bar_next).sqrt() * g_init_theta
            else:
                raise ValueError(f'Unknown sampling method: {sampling}')

            if torch.isnan(x_init_next).any():
                print('nan:', t)
            x_init_t_list.append(x_init_next.detach())
            if not if_output_diffusion_process:
                x_init_t_list.pop(0)
            
        output = x_init_t_list if if_output_diffusion_process else x_init_t_list[-1]
        return output, wgt_store

    
    def sample_guide(self, 
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling=None,
               stride=20,
               grad_guid = None,
               guid_param = None,
               ) -> Dict[str, torch.Tensor]:
        task = guid_param['task']
        cost_param = guid_param['cost_param']
        cost_param_costl = cost_param['cost_param_costl']
        cost_param_threl = cost_param['cost_param_threl']
        guid_label, latent_mean, latent_std = grad_guid
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps
        
        device = scene_enc['x_pl'].device
        
        num_agents = eval_mask.sum()

        e_init_rand = torch.randn([num_agents,  self.net.m_dim]).to(device)
        
        
        s_T = torch.sqrt(self.var_sched.alpha_bars[reverse_steps].to(device))
        if start_data is None:
            c1 = 1
            x_init_T = c1 * e_init_rand + s_T
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)

            if start_data.dim() == 2:
                x_init_T = c0 * start_data.unsqueeze(1) + c1 * e_init_rand
            elif start_data.dim() == 3:
                x_init_T = c0 * start_data + c1 * e_init_rand
        
        x_init_t_list = [x_init_T]
        
        torch.cuda.empty_cache()
        
        wgt_store = []
        for t in range(reverse_steps, 0, -stride):

            beta = self.var_sched.betas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            
            
            x_init_t = x_init_t_list[-1]
            
            
            with torch.no_grad():
                beta_emb = beta.unsqueeze(-1).repeat(num_agents, 1).to(device)
                g_init_theta, wgts = self.net(copy.deepcopy(x_init_t), beta_emb, data, scene_enc, eval_mask=eval_mask, mode=0)     
                if t in [10, 100, 200, 300, 400]:
                    wgt_store.append(wgts)

            with torch.inference_mode(False):
                temp_x_0 = (x_init_t - torch.sqrt(1-alpha_bar).clone() * g_init_theta) / torch.sqrt(alpha_bar).clone()


                temp_x_0 = temp_x_0.clone().detach().requires_grad_(True)
                
                diff = self.task_diff(
                    task, temp_x_0, guid_label,
                    latent_mean.clone().detach(), latent_std.clone().detach())
                error = diff[:1].sum() * 0.1 + diff[1:].sum()
                error.backward()
                # note it is the gradient of x0
                grad = temp_x_0.grad      
                grad = grad * cost_param_costl
                
            grad = torch.clamp(grad, min=-cost_param_threl, max=cost_param_threl)
            m_init_0 = temp_x_0 - grad
            
            m_init_0 = torch.clamp(m_init_0, min=-3.0, max=3.0)
            x_init_next = alpha_bar_next.sqrt() * m_init_0 + (1 - alpha_bar_next).sqrt() * g_init_theta

            if torch.isnan(x_init_next).any():
                print('nan:', t)
            x_init_t_list.append(x_init_next.detach())
            
            if not if_output_diffusion_process:
                x_init_t_list.pop(0)
        
            
        output = x_init_t_list if if_output_diffusion_process else x_init_t_list[-1]
        return output, wgt_store

class HLDenoiser(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 init_timestep: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 m_dim: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.init_timestep = init_timestep
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.m_dim = m_dim

        self.proj_in_m_delta = nn.Linear(m_dim//2, hidden_dim)
        self.proj_in_m_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU() # Or ReLU
        )
        self.proj_out_m_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.proj_out_m_speed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.num_future_steps = 2
        noise_dim = 1
        self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.type_a_emb = nn.Embedding(10, hidden_dim)

        self.timestep_emb = nn.Embedding(self.num_future_steps, hidden_dim)
        self.agent_id_emb = nn.Embedding(1000, hidden_dim)
        ########
        
        self.interact_pl2m = nn.ModuleList(
            [TransformerDecoderLayerDiff(
            n_embd=hidden_dim,
            n_head=num_heads,
            ff_dim=4 * hidden_dim,
            dropout=dropout,
            layer_id=i,
        )  for i in range(num_layers)])        
        
        self.to_out_m = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.1), # LeakyReLU helps prevent dead neurons in regression
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim) 
            # No activation on the very last layer for regression!
        )

        probs = torch.tensor([0.5])
        self.B_dist = Bernoulli(probs=probs)

        self.apply(weight_init)
    
    def forward(self,
                m_delta,
                beta,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                eval_mask,
                mode=0
     ) -> Dict[str, torch.Tensor]:

        device = m_delta.device
        
        x_pl = scene_enc['x_pl']
        map_summary = scene_enc['map_summary']

        map_batch_list = data['map_polygon']['batch']
        batch_size = data.num_graphs
        poly_cnt_per_batch = map_batch_list.bincount(minlength=batch_size)
        map_emb_batch = torch.split(x_pl, poly_cnt_per_batch.tolist())

        map_emb = pad_sequence(map_emb_batch, batch_first=True, padding_value=0)         
        beta_emb = self.noise_emb(beta)
        categorical_embs_m = [
                self.type_a_emb(data['agent']['type'][eval_mask].long()),
            ]

        
        agent_batch_list = data['agent']['batch'][data['agent']['mask']]
        agent_cnt_per_batch = agent_batch_list.bincount(minlength=batch_size)
        agent_ids = torch.cat([torch.arange(cnt) for cnt in agent_cnt_per_batch ]).to(device)
        m_init = m_delta[..., :self.m_dim//2]
        m_final = m_delta[..., self.m_dim//2:]

        m_init = self.proj_in_m_delta(m_init)
        m_final = self.proj_in_m_delta(m_final)

        m_delta = torch.stack([m_init, m_final], dim=1)
        agent_id_emb = self.agent_id_emb(agent_ids)
        timestep_emb = self.timestep_emb(torch.arange(2).to(device))

        cat_emb = categorical_embs_m[0].unsqueeze(1).expand(-1, 2, -1)
        id_emb = agent_id_emb.unsqueeze(1).expand(-1, 2, -1)
        time_emb = timestep_emb.unsqueeze(0).expand(m_delta.size(0), -1, -1)
        m_delta = torch.cat([m_delta, cat_emb, id_emb, time_emb], dim=-1)
        m_delta = self.proj_in_m_mlp(m_delta)

        agent_emb_batch = torch.split(m_delta, agent_cnt_per_batch.tolist())
        m = pad_sequence(agent_emb_batch, batch_first=True, padding_value=0)

        beta_emb_batch = torch.split(beta_emb, agent_cnt_per_batch.tolist())
        beta_emb_m = pad_sequence(beta_emb_batch, batch_first=True, padding_value=0).unsqueeze(2)#repeat(1, 2, 1)

        B, N, T, D = m.shape
        B, N_map, _ = map_emb.shape
        
        attn_mask_agent_layers = torch.ones(B, N, device=device, dtype=torch.bool)
        for i, cnt in enumerate(agent_cnt_per_batch):
            attn_mask_agent_layers[i, :cnt] = False

        attn_mask_map_layers = torch.ones(B, N_map, device=device, dtype=torch.bool)
        for i, cnt in enumerate(poly_cnt_per_batch):
            attn_mask_map_layers[i, :cnt] = False                      
        attn_mask_map_layers= attn_mask_map_layers.view(B, 1, N_map)
        
        attn_mask_agent_layers = attn_mask_agent_layers.unsqueeze(2) | attn_mask_agent_layers.unsqueeze(1)
        attn_mask_agent_layers = attn_mask_agent_layers.unsqueeze(1).repeat(1,2,1, 1).reshape(B*T, N, N)
        
        if mode:            
            I_mat = torch.eye(N, device=device, dtype=torch.bool)   
            attn_mask_agent_layers = attn_mask_agent_layers + ~I_mat.unsqueeze(0).expand(B*T, -1, -1)

        m_sum = m
        wgts = []
        for i in range(self.num_layers):
            m = m + beta_emb_m

            m, map2agent_wgts = self.interact_pl2m[i](x=m, map_enc=map_emb,
                                            mask=attn_mask_agent_layers,
                                            map_mask=attn_mask_map_layers, 
                                            map_summary=map_summary)
            wgts.append(map2agent_wgts)
            m_sum = m_sum + m

        mask = torch.arange(N).expand(B, N).to(device) < agent_cnt_per_batch.unsqueeze(1)  # [B, N]
        mask_agent = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)  # [B, N, D]
        m_sum = m_sum[mask_agent].view(-1, T, D)  # [sum(agent_cnt_per_batch), D]
        m_out = self.to_out_m(m_sum.view(-1, T*D))
        m_out_pos = self.proj_out_m_pos(m_out)

        m_out_speed = self.proj_out_m_speed(m_out)
        eps_init = torch.cat([m_out_pos[...,:2], m_out_speed[...,:1]], dim=-1)
        eps_goal = torch.cat([m_out_pos[...,2:], m_out_speed[...,1:]], dim=-1)
        out = torch.cat([eps_init, eps_goal], dim=-1)

        return out, wgts


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        
        alphas = 1 - betas
        
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        
        # kt
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        kt = 1 - sqrt_alpha_bars # shifted diffusion
        self.register_buffer('kt', kt)
        
        inv_sqrt_alpha = 1 / torch.sqrt(alphas)
        co_g = betas / torch.sqrt(1-alpha_bars)
        co_st = torch.sqrt(alphas[1:]) * (1-alpha_bars[:-1])/(1-alpha_bars[1:])
        co_st = torch.cat([torch.tensor([0]),co_st])
        co_z = torch.sqrt((1-alpha_bars[:-1])/(1-alpha_bars[1:])*betas[1:])
        co_z = torch.cat([torch.tensor([0]),co_z])
        self.register_buffer('inv_sqrt_alpha', inv_sqrt_alpha)
        self.register_buffer('co_g', co_g)
        self.register_buffer('co_st', co_st)
        self.register_buffer('co_z', co_z)
        

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
