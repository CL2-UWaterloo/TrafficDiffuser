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

import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_cluster import radius
import pytorch_lightning as pl


from metrics import  OffRoad, Collision, NearestEdge
from metrics import JSD_SPEED, JSD_MAP_DIST, JSD_INTERACTIVE, JSD_LOCAL_DENSITY

from module_hl_generation import TDMapEncoder, HLDiffusion

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

from visualization import *
from utils import denormalize

class HLTrafficGen(pl.LightningModule):

    def __init__(self,
                 args,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.output_head = args.output_head
        self.init_timestep = args.init_timestep
        self.num_freq_bands = args.num_freq_bands
        self.num_map_layers = args.num_map_layers
        self.num_dec_layers = args.num_dec_layers
        self.num_heads = args.num_heads

        self.head_dim = args.head_dim
        self.dropout = args.dropout
        self.pl2pl_radius = args.pl2pl_radius

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.T_max = args.T_max

        self.sampling = args.sampling
        self.sampling_stride = args.sampling_stride
        self.num_diffusion_steps = args.num_diffusion_steps

        self.m_dim = args.m_dim
        self.root = args.root
        self.T = 2
        
        self.check_param()

        stats = torch.load('state_stats.pt')
        self.m_mean = stats['mean']
        self.m_std = stats['std']

        self.td_mapencoder = TDMapEncoder(dataset=args.dataset,
                                                input_dim=self.input_dim,
                                                hidden_dim=self.hidden_dim,
                                                pl2pl_radius=self.pl2pl_radius,
                                                num_freq_bands=self.num_freq_bands,
                                                num_layers=self.num_map_layers,
                                                num_heads=self.num_heads,
                                                head_dim=self.head_dim,
                                                dropout=self.dropout,
                                                state_stats=stats)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.hl_diffusion = HLDiffusion(args=args)

        self.OffRoad =  nn.ModuleList([OffRoad() for _ in range(self.T)])
        self.Collision = nn.ModuleList([Collision() for _ in range(self.T)])
        self.NearestEdge = nn.ModuleList([NearestEdge() for _ in range(self.T)])

        self.OffRoad_gt = nn.ModuleList([OffRoad() for _ in range(self.T)])
        self.Collision_gt = nn.ModuleList([Collision() for _ in range(self.T)])
        self.NearestEdge_gt = nn.ModuleList([NearestEdge() for _ in range(self.T)])

        self.JSD_MAP_DIST = nn.ModuleList([JSD_MAP_DIST() for _ in range(self.T)])
        self.JSD_LOCAL_DENSITY = nn.ModuleList([JSD_LOCAL_DENSITY() for _ in range(self.T)])
        self.JSD_INTERACTIVE = nn.ModuleList([JSD_INTERACTIVE() for _ in range(self.T)])
        self.JSD_SPEED = nn.ModuleList([JSD_SPEED() for _ in range(self.T)])

    def add_extra_param(self, args):
        self.guid_task = args.guid_task
        self.hl_diffusion.guid_task = args.guid_task

        self.cond_norm = args.cond_norm
        self.cost_param_costl = args.cost_param_costl
        self.cost_param_threl = args.cost_param_threl

        self.plot = args.plot
        self.save_diffusion_steps = args.save_diffusion_steps
        self.root = args.root
        if hasattr(args, 'hl_ckpt_path'):
            self.ckpt_path = args.hl_ckpt_path
        else:
            self.ckpt_path = None
        
        self.save_dir = 'hl'
        if self.ckpt_path:
            self.save_dir += '_' + self.ckpt_path.split('/')[-5]

    def check_param(self):
        if self.sampling == 'ddpm':
            self.sampling_stride = 1
        elif self.sampling == 'ddim':
            self.sampling_stride = int(self.sampling_stride)
            if self.sampling_stride > self.num_diffusion_steps - 1:
                print('ddim stride > diffusion steps.')
                exit()
            scale = self.num_diffusion_steps / self.sampling_stride
            if abs(scale - int(scale)) > 0.00001:
                print('mod(diffusion steps, ddim stride) != 0')
                exit()

    def forward(self, data: HeteroData):
        scene_enc = self.td_mapencoder(data)
        x = torch.ones(32,10).to(scene_enc['x_a'].device)
        return self.linear(x)

    def _guidance_args(self, data):
        if self.guid_task == 'none':
            return {}
        return {
            'grad_guid': [data, self.m_mean, self.m_std],
            'guid_param': {
                'task': self.guid_task,
                'cost_param': {
                    'cost_param_costl': self.cost_param_costl,
                    'cost_param_threl': self.cost_param_threl,
                },
            },
        }

    @staticmethod
    def nearest_map_heading(positions, data):
        map_positions = data['map_point']['position'][..., :2]
        max_neighbors = data['map_point']['batch'].bincount().max().item()
        edges = radius(x=positions, y=map_positions, r=4,
                       max_num_neighbors=max_neighbors)
        headings = positions.new_zeros(positions.shape[0])
        distances = torch.norm(positions[edges[1]] - map_positions[edges[0]], dim=-1)
        for agent_index in edges[1].unique():
            candidates = torch.where(edges[1] == agent_index)[0]
            nearest = edges[0][candidates[torch.argmin(distances[candidates])]]
            headings[agent_index] = data['map_point']['orientation'][nearest]
        return headings

    def generate_hl_scenarios(self, data, eval_mask=None):
        """Generate endpoint states in the representation expected by TrajInfill."""
        if eval_mask is None:
            eval_mask = data['agent']['mask']
        scene_enc = self.td_mapencoder(data)
        self.load_vars(data['agent']['position'].device)
        agent_batch = data['agent']['batch'][eval_mask]
        map_min = data['map_min'].view(-1, 3)[..., :2][agent_batch]
        map_max = data['map_max'].view(-1, 3)[..., :2][agent_batch]

        sampled, _ = self.hl_diffusion.sample(
            data=data, scene_enc=scene_enc, sampling=self.sampling,
            stride=self.sampling_stride, eval_mask=eval_mask,
            if_output_diffusion_process=self.save_diffusion_steps, reverse_steps=None,
            **self._guidance_args(data))

        if self.save_diffusion_steps:
            latent = sampled[-1]
            intermediate = []
            for step_latent in sampled:
                step_init, _ = denormalize(
                    step_latent[..., :3], self.m_mean, self.m_std, map_min, map_max)
                step_final, _ = denormalize(
                    step_latent[..., 3:], self.m_mean, self.m_std, map_min, map_max)
                intermediate.append(torch.stack((step_init, step_final), dim=1))
            self.intermediate_endpoints = torch.stack(intermediate).detach().cpu().numpy()

        else:
            latent = sampled
            intermediate = []

        init_pos, init_speed = denormalize(
            latent[..., :3], self.m_mean, self.m_std, map_min, map_max)
        final_pos, final_speed = denormalize(
            latent[..., 3:], self.m_mean, self.m_std, map_min, map_max)

        if torch.isnan(init_pos).any() or torch.isnan(final_pos).any():
            raise RuntimeError('HL endpoint sampling produced NaN positions')

        init_state = torch.cat((init_pos, self.nearest_map_heading(init_pos, data)[:, None],
                                init_speed[:, None]), dim=-1)
        final_state = torch.cat((final_pos, self.nearest_map_heading(final_pos, data)[:, None],
                                 final_speed[:, None]), dim=-1)
        return init_state, final_state

    def training_step(self,
                      data,
                      batch_idx):
        should_print = batch_idx % 100 == 0
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        self.load_vars(self.device_)
        eval_mask = data['agent']['mask']
        scene_encoding = self.td_mapencoder(data)

        positions = data['agent']['scaled_position'][eval_mask, self.init_timestep:]
        speeds = data['agent']['scaled_speed'][eval_mask, self.init_timestep:]

        initial_endpoint = torch.cat((positions[:, 0], speeds[:, :1]), dim=-1)
        final_endpoint = torch.cat((positions[:, -1], speeds[:, -1:]), dim=-1)
        initial_endpoint = (initial_endpoint - self.m_mean) / self.m_std
        final_endpoint = (final_endpoint - self.m_mean) / self.m_std
        diffusion_target = torch.cat((initial_endpoint, final_endpoint), dim=-1)

        diffusion_loss, _, entropy = self.hl_diffusion.get_loss(
            diffusion_target,
            data=data,
            scene_enc=scene_encoding,
            eval_mask=eval_mask,
        )

        diffusion_loss = diffusion_loss.mean()
        loss = diffusion_loss + 0.1 * entropy

        log_options = dict(
            prog_bar=False, on_step=True, on_epoch=True,
            batch_size=1, sync_dist=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], **log_options)
        self.log('train/loss_diff', diffusion_loss, **log_options)
        self.log('train/entropy_mean', entropy, **log_options)

        if should_print:
            print(
                f'{batch_idx}, loss_diff: {diffusion_loss.item():.5f}, '
                f'entropy_mean: {entropy.item():.5f}')

        return loss

    def validation_step(self,
                    data,
                    batch_idx):
        print_flag = False
        if batch_idx % 100 == 0:
            print_flag = True
            
        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        eval_mask = data['agent']['mask']
        

        agent_batch = data['agent']['batch'][eval_mask]
        init_state, final_state = self.generate_hl_scenarios(data, eval_mask)
        pred_trans_init, pred_trans_final = init_state[:, :2], final_state[:, :2]
        pred_speed_init, pred_speed_final = init_state[:, 3], final_state[:, 3]

        origin_init_eval = data['agent']['position'][eval_mask, self.init_timestep, :2]
        origin_final_eval = data['agent']['position'][eval_mask, -1, :2]
        gt_trans =  torch.stack([origin_init_eval, origin_final_eval], dim=1)
        gt_speed = torch.norm(data['agent']['velocity'], dim=-1)
        gt_speed = torch.stack([gt_speed[eval_mask, self.init_timestep],
                                    gt_speed[eval_mask, -1]], dim=1)
    
        pred_trans =  torch.stack([pred_trans_init, pred_trans_final], dim=1)
        pred_speed = torch.stack([pred_speed_init, pred_speed_final], dim=1)
        trans_loss = torch.nn.MSELoss()(pred_trans, gt_trans)

        self.log('val_trans_loss', trans_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        self.log(f'val_offroad_rate', self.OffRoad[0], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

        for i in range(self.T):
            self.JSD_LOCAL_DENSITY[i].update(pred=pred_trans[:, i], gt=gt_trans[:, i], agent_batch=agent_batch, data=data)
            self.log(f'val/jsd_local_density_{i}', self.JSD_LOCAL_DENSITY[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
            self.JSD_INTERACTIVE[i].update(pred_trans[:, i], gt_trans[:, i], agent_batch=agent_batch)
            self.log(f'val/jsd_interactive_{i}', self.JSD_INTERACTIVE[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            self.JSD_MAP_DIST[i].update(pred=pred_trans[:, i], gt=gt_trans[:, i], agent_batch=agent_batch, map_pts=data['map_point'])
            self.log(f'val/jsd_map_dist_{i}', self.JSD_MAP_DIST[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            self.JSD_SPEED[i].update(pred_speed[:, i], gt_speed[:, i])
            self.log(f'val/jsd_speed_{i}', self.JSD_SPEED[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            self.OffRoad[i].update(pred=pred_trans[:, i], agent_batch=agent_batch, map_pts=data['map_point'])
            self.log(f'val/offroad_rate_{i}', self.OffRoad[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
            self.NearestEdge[i].update(pred=pred_trans[:, i], agent_batch=agent_batch, map_pts=data['map_point'])
            self.log(f'val/nearest_edge_dist_{i}', self.NearestEdge[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
            self.Collision[i].update(pred=pred_trans[:, i], agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
            self.log(f'val/collision_rate_{i}', self.Collision[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            self.OffRoad_gt[i].update(pred=gt_trans[:, i],agent_batch=agent_batch, map_pts=data['map_point'])
            self.log(f'val/offroad_rate_gt_{i}', self.OffRoad_gt[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            self.NearestEdge_gt[i].update(pred=gt_trans[:, i],agent_batch=agent_batch, map_pts=data['map_point'])
            self.log(f'val/nearest_edge_dist_gt_{i}', self.NearestEdge_gt[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            self.Collision_gt[i].update(pred=gt_trans[:, i], agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
            self.log(f'val/collision_rate_gt_{i}', self.Collision_gt[i], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)

            if print_flag:
                print(f'Timestep {i}:')
                print(f'GT: collision: {self.Collision_gt[i].compute().item()}, nearest_edge:{self.NearestEdge_gt[i].compute().item()}, offroad:{self.OffRoad_gt[i].compute().item()}')
                print(f'Gen: collision: {self.Collision[i].compute().item()}, nearest_edge:{self.NearestEdge[i].compute().item()}, offroad:{self.OffRoad[i].compute().item()}')
        

        if self.plot or self.save_diffusion_steps:
            img_folder = 'visual'
            sub_folder = self.save_dir
            num_scenes = agent_batch[-1].item()+1
            num_agents_per_scene = pred_trans_init.new_tensor([(agent_batch == i).sum() for i in range(num_scenes)]).type(torch.int64)

            for i in range(num_scenes):
                start_id = int(torch.sum(num_agents_per_scene[:i]).item())
                end_id = int(torch.sum(num_agents_per_scene[:i+1]).item())
                
                gt_eval_world = gt_trans.detach().cpu().numpy()
                rec_eval_world = pred_trans.unsqueeze(1).detach().cpu().numpy()
                if end_id - start_id == 1:
                    continue
            
                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path(f'{self.root}/val/raw')
                scenario_folder = base_path_to_data / scenario_id
                
                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"

                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)
                
                viz_output_dir = Path(img_folder) / sub_folder
                os.makedirs(viz_output_dir,exist_ok=True)

                viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'.svg')
                
                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['gen_hl'] = rec_eval_world[start_id:end_id]

                traj_visible = {}
                traj_visible['gt'] = False
                traj_visible['gen_hl'] = True

                visualize_hl_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path, data)
                if self.save_diffusion_steps:
                    for j, traj in enumerate(self.intermediate_endpoints):
                        additional_traj['gen_hl'] = traj[start_id:end_id, np.newaxis]
                        viz_save_path = viz_output_dir / (
                            f'b{data_batch}_s{i}_{self.sampling}_step_{j:04d}.svg')
                        visualize_hl_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path, data)

        
    def load_vars(self, device):
        if  self.m_mean.device != device:
            self.m_mean =  self.m_mean.to(device)
            self.m_std =  self.m_std.to(device)


    def on_before_optimizer_step(self, optimizer):
        # Calculate total gradient norm
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if self.global_step % 1000 == 0:
            print(f"Step {self.global_step}: Gradient Norm = {total_norm:.4f}")
        self.log("global_grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True)            
        
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, weight_decay=self.weight_decay, lr=self.lr)

        max_steps = self.trainer.estimated_stepping_batches
            
        T_mult = 2
        T_0 = 100
        num_cycles = max(
            1,
            math.floor(
                math.log2(max_steps / T_0 + 1)
            ),
        )

        num_restart_cycles = max(1, num_cycles - 1)
        restart_steps = T_0 * (T_mult**num_restart_cycles - 1)
        final_steps = max_steps - restart_steps

        warm_restart = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-7)
        final_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=final_steps,
            eta_min=1e-7,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warm_restart,
                final_cosine,
            ],
            milestones=[
                restart_steps,
            ],
        )

        return [optimizer], [{
        'scheduler': scheduler,
        'interval': 'step',  # or 'epoch', depending on when you want to step the scheduler
        'frequency': 1
        }]


    @staticmethod
    def add_model_specific_args(parent_parser, training=True, prefix="hl_"):
        parser = parent_parser.add_argument_group('TrafficDiffuser_HL')
        
        parser.add_argument(f'--{prefix}input_dim', dest=f'{prefix}input_dim', type=int, default=2)
        parser.add_argument(f'--{prefix}hidden_dim', dest=f'{prefix}hidden_dim', type=int, default=128)
        parser.add_argument(f'--{prefix}output_dim', dest=f'{prefix}output_dim', type=int, default=2)
        parser.add_argument(f'--{prefix}output_head', dest=f'{prefix}output_head', action='store_true')
        parser.add_argument(f'--{prefix}init_timestep', dest=f'{prefix}init_timestep', type=int, required=training)
        parser.add_argument(f'--{prefix}num_freq_bands', dest=f'{prefix}num_freq_bands', type=int, default=64)
        parser.add_argument(f'--{prefix}num_map_layers', dest=f'{prefix}num_map_layers', type=int, default=1)
        parser.add_argument(f'--{prefix}num_dec_layers', dest=f'{prefix}num_dec_layers', type=int, default=2)
        parser.add_argument(f'--{prefix}num_heads', dest=f'{prefix}num_heads', type=int, default=8)
        parser.add_argument(f'--{prefix}head_dim', dest=f'{prefix}head_dim', type=int, default=16)
        parser.add_argument(f'--{prefix}dropout', dest=f'{prefix}dropout', type=float, default=0.1)
        parser.add_argument(f'--{prefix}pl2pl_radius', dest=f'{prefix}pl2pl_radius', type=float, required=training)
        parser.add_argument(f'--{prefix}lr', dest=f'{prefix}lr', type=float, default=5e-4)
        parser.add_argument(f'--{prefix}weight_decay', dest=f'{prefix}weight_decay', type=float, default=1e-4)
        parser.add_argument(f'--{prefix}T_max', dest=f'{prefix}T_max', type=int, default=64)
        parser.add_argument(f'--{prefix}num_denoiser_layers', dest=f'{prefix}num_denoiser_layers', type=int, default=3)
        parser.add_argument(f'--{prefix}num_diffusion_steps', dest=f'{prefix}num_diffusion_steps', type=int, default=10)
        parser.add_argument(f'--{prefix}beta_1', dest=f'{prefix}beta_1', type=float, default=1e-4)
        parser.add_argument(f'--{prefix}beta_T', dest=f'{prefix}beta_T', type=float, default=0.05)
        parser.add_argument(f'--{prefix}sampling', dest=f'{prefix}sampling', choices=['ddpm','ddim'])
        parser.add_argument(f'--{prefix}sampling_stride', dest=f'{prefix}sampling_stride', type = int, default = 20)
        parser.add_argument(f'--{prefix}save_diffusion_steps', dest=f'{prefix}save_diffusion_steps',
                            action='store_true')
        parser.add_argument(f'--{prefix}guid_task', dest=f'{prefix}guid_task', choices=['none', 'map', 'map_collision', 'original'],default = 'none')
        parser.add_argument(f'--{prefix}cond_norm', dest=f'{prefix}cond_norm', type = int, default = 0)
        parser.add_argument(f'--{prefix}cost_param_costl', dest=f'{prefix}cost_param_costl', type = float, default = 1.0)
        parser.add_argument(f'--{prefix}cost_param_threl', dest=f'{prefix}cost_param_threl', type = float, default = 1.0)
        parser.add_argument(f'--{prefix}m_dim', dest=f'{prefix}m_dim', type = int,default = 6)
        
        return parent_parser
