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

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

import pytorch_lightning as pl

from pathlib import Path

from module_trajectory_infill import QCNetMapEncoder, Infiller

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

from metrics import MR
from metrics import minADE
from metrics import minFDE

from visualization import *
from utils import rotation_matrix

class TrajInfill(pl.LightningModule):

    def __init__(self,
                 args,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.init_timestep = args.init_timestep
        self.num_infill_steps = args.num_infill_steps

        self.num_freq_bands = args.num_freq_bands
        self.num_map_layers = args.num_map_layers
        self.num_dec_layers = args.num_dec_layers
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.dropout = args.dropout
        self.pl2pl_radius = args.pl2pl_radius
        self.a2a_radius = args.a2a_radius
        self.pl2m_radius = args.pl2m_radius
        self.a2m_radius = args.a2m_radius
        self.lr = args.lr
        self.weight_decay = args.weight_decay


        self.root = args.root

        self.qcnet_mapencoder = QCNetMapEncoder(dataset=args.dataset,
                                                input_dim=self.input_dim,
                                                hidden_dim=self.hidden_dim,
                                                init_timestep=0,
                                                pl2pl_radius=self.pl2pl_radius,
                                                num_freq_bands=self.num_freq_bands,
                                                num_layers=self.num_map_layers,
                                                num_heads=self.num_heads,
                                                head_dim=self.head_dim,
                                                dropout=self.dropout)
        self.infiller = Infiller(
            dataset=args.dataset,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            init_timestep=self.init_timestep,
            num_infill_steps=self.num_infill_steps,
            pl2m_radius=self.pl2m_radius,
            a2m_radius=self.a2m_radius,
            num_freq_bands=self.num_freq_bands,
            num_layers=self.num_dec_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout,
        )

        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mse_loss = nn.MSELoss(reduction='none')      

        self.minADE = minADE(max_guesses=1)
        self.minFDE = minFDE(max_guesses=1)
        self.MR = MR(max_guesses=1)

    def add_extra_param(self, args):

        self.plot = args.plot
        self.root = args.root
        self.data_split = getattr(args, 'network_mode', 'val')

        if hasattr(args, 'infill_ckpt_path'):
            self.ckpt_path = args.infill_ckpt_path
        else:
            self.ckpt_path = None
            
        self.save_dir = 'infill'
        if self.ckpt_path:
            self.save_dir += '_' + self.ckpt_path.split('/')[-5]

    def forward(self, data: HeteroData):
        scene_enc = self.qcnet_mapencoder(data)
        x = torch.ones(32,10).to(scene_enc['x_a'].device)
        return self.linear(x)

    def ground_truth_endpoints(self, data, eval_mask=None):
        """Build endpoint states in the representation used by the infiller."""
        if eval_mask is None:
            eval_mask = data['agent']['mask']
        positions = data['agent']['position'][eval_mask, self.init_timestep:, :self.input_dim]
        headings = data['agent']['heading'][eval_mask, self.init_timestep:]
        speeds = torch.norm(
            data['agent']['velocity'][eval_mask, self.init_timestep:], p=2, dim=-1) / 15 - 1
        init_state = torch.cat((positions[:, 0], headings[:, :1], speeds[:, :1]), dim=-1)
        final_state = torch.cat((positions[:, -1], headings[:, -1:], speeds[:, -1:]), dim=-1)
        return init_state, final_state

    def infill_trajectory(self, data, init_state, final_state, mode=0):
        """Infill a local trajectory between supplied endpoint states."""
        scene_enc = self.qcnet_mapencoder(data)
        if torch.is_tensor(mode):
            modes = mode.to(device=init_state.device, dtype=torch.long)
        else:
            modes = torch.full((init_state.shape[0],), mode, device=init_state.device,
                               dtype=torch.long)
        return self.infiller(data, scene_enc, init_state, final_state, mode=modes)

    def trajectory_loss(self, prediction, target, valid_mask):
        loss = self.mse_loss(prediction.squeeze(1), target).sum(dim=-1) * valid_mask
        return (loss.sum(dim=0) / valid_mask.sum(dim=0).clamp(min=1)).mean()

    def trajectory_to_world(self, trajectory, init_state):
        """Convert a local infilled trajectory to world coordinates."""
        rotation = rotation_matrix(init_state[:, 2]).transpose(1, 2)
        world = torch.matmul(trajectory.squeeze(1)[..., :2], rotation)
        return (world + init_state[:, None, :2]).unsqueeze(1)

    def _visualize(self, data, batch_idx, eval_mask, ground_truth, prediction):

        ground_truth = ground_truth.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        scene_index = data['agent']['batch'][eval_mask]
        num_scenes = int(scene_index[-1]) + 1
        output_dir = Path('visual') / self.save_dir
        os.makedirs(output_dir, exist_ok=True)

        for scene_i in range(num_scenes):
            selected = torch.where(scene_index == scene_i)[0].cpu().numpy()
            if selected.size <= 1:
                continue
            scenario_id = data['scenario_id'][scene_i]
            scenario_dir = Path(self.root) / self.data_split / 'raw' / scenario_id
            scenario = scenario_serialization.load_argoverse_scenario_parquet(
                scenario_dir / f'scenario_{scenario_id}.parquet')
            static_map = ArgoverseStaticMap.from_json(
                scenario_dir / f'log_map_archive_{scenario_id}.json')
            rank = getattr(self.trainer, 'global_rank', 0)
            save_path = output_dir / f'r{rank}_b{batch_idx}_s{scene_i}.svg'
            additional_traj = {
                'gt_init': ground_truth[selected],
                'infilled': prediction[selected, None],
            }
            traj_visible = {'gt': False, 'infilled': True}
            visualize_scenario_infilling_prediction(
                scenario, static_map, additional_traj, traj_visible, save_path, data)

    def training_step(self,
                      data,
                      batch_idx):
        should_print = batch_idx % 100 == 0
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        eval_mask = data['agent']['mask']
        valid_mask = data['agent']['predict_mask'][eval_mask, self.init_timestep:]
        target = data['agent']['target'][eval_mask, :, :self.output_dim]
        initial_endpoint, final_endpoint = self.ground_truth_endpoints(data, eval_mask)

        mode_probabilities = initial_endpoint.new_full((4,), 0.25)
        modes = torch.multinomial(
            mode_probabilities, num_samples=initial_endpoint.shape[0], replacement=True)
        prediction = self.infill_trajectory(
            data, initial_endpoint, final_endpoint, mode=modes)
        trajectory = prediction[..., :self.output_dim]

        regression_loss = self.trajectory_loss(trajectory, target, valid_mask)
        trajectory_delta = trajectory[:, :, 1:] - trajectory[:, :, :-1]
        smoothing_loss = torch.norm(trajectory_delta, p=2, dim=-1).mean(dim=-1) * valid_mask[:, 1:]
        smoothing_loss = (smoothing_loss.sum(dim=0) / valid_mask[:, 1:].sum(dim=0).clamp(min=1)).mean()
        loss = regression_loss + 0.5 * smoothing_loss

        log_options = dict(
            prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_propose', regression_loss, **log_options)
        self.log('train_smoothing_loss', smoothing_loss, **log_options)

        if should_print:
            print(
                f'Training Batch {batch_idx}: '
                f'Reg Loss Propose: {regression_loss.item():.4f}, '
                f'smoothing_loss: {smoothing_loss.item():.4f}')

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

        reg_mask = data['agent']['predict_mask'][:, self.init_timestep:]
        reg_mask = reg_mask[eval_mask]

        gt_eval = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)[eval_mask]
        gt_init, gt_final = self.ground_truth_endpoints(data, eval_mask)
        pred = self.infill_trajectory(data, gt_init, gt_final)

        traj_propose = pred[..., :self.output_dim]

        reg_loss_propose = self.trajectory_loss(
            traj_propose[..., :self.output_dim], gt_eval[..., :self.output_dim], reg_mask)

        self.log('val/reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        valid_mask_eval = reg_mask
        traj_eval = traj_propose[..., :self.output_dim] 
        
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                           valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                            valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim],
                       valid_mask=valid_mask_eval)
        self.log('val/minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val/minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val/MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        if print_flag:
            print(f"Validation Batch {batch_idx}: "
                  f"Reg Loss Propose: {reg_loss_propose.item():.4f}, "
                  f"minADE: {self.minADE.compute().item():.4f}, "
                  f"minFDE: {self.minFDE.compute().item():.4f}, "
                  f"MR: {self.MR.compute().item():.4f}")


        gt_xy = gt_eval[..., :2]

        pred_xy = self.trajectory_to_world(traj_eval, gt_init).squeeze(1)
        gt_xy = self.trajectory_to_world(gt_xy.unsqueeze(1), gt_init).squeeze(1)
        if self.plot:
            self._visualize(data, data_batch, eval_mask, gt_xy, pred_xy)




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
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.lr, steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,  # Or len(train_dataloader) if you know it
            epochs=self.trainer.max_epochs, pct_start=0.05)
                
        return [optimizer], [{
        'scheduler': scheduler,
        'interval': 'step',  # or 'epoch', depending on when you want to step the scheduler
        'frequency': 1
        }]

#   f"--{prefix}hidden_dim",
#             dest=f"{prefix}hidden_dim",
    @staticmethod
    def add_model_specific_args(parent_parser, training=True, prefix="infill_"):
        parser = parent_parser.add_argument_group('TrafficDiffuser_infill')
        parser.add_argument(f'--{prefix}input_dim', dest=f'{prefix}input_dim', type=int, default=2)
        parser.add_argument(f'--{prefix}hidden_dim', dest=f'{prefix}hidden_dim', type=int, default=128)
        parser.add_argument(f'--{prefix}output_dim', dest=f'{prefix}output_dim', type=int, default=2)
        parser.add_argument(f'--{prefix}init_timestep', dest=f'{prefix}init_timestep', type=int, required=training)
        parser.add_argument(f'--{prefix}num_infill_steps', dest=f'{prefix}num_infill_steps', type=int, required=training)
        parser.add_argument(f'--{prefix}num_freq_bands', dest=f'{prefix}num_freq_bands', type=int, default=64)
        parser.add_argument(f'--{prefix}num_map_layers', dest=f'{prefix}num_map_layers', type=int, default=1)
        parser.add_argument(f'--{prefix}num_dec_layers', dest=f'{prefix}num_dec_layers', type=int, default=2)
        parser.add_argument(f'--{prefix}num_heads', dest=f'{prefix}num_heads', type=int, default=8)
        parser.add_argument(f'--{prefix}head_dim', dest=f'{prefix}head_dim', type=int, default=16)
        parser.add_argument(f'--{prefix}dropout', dest=f'{prefix}dropout', type=float, default=0.1)

        parser.add_argument(f'--{prefix}pl2pl_radius', dest=f'{prefix}pl2pl_radius', type=float, required=training)
        parser.add_argument(f'--{prefix}a2a_radius', dest=f'{prefix}a2a_radius', type=float, required=training)
        parser.add_argument(f'--{prefix}pl2m_radius', dest=f'{prefix}pl2m_radius', type=float, required=training)
        parser.add_argument(f'--{prefix}a2m_radius', dest=f'{prefix}a2m_radius', type=float, required=training)

        parser.add_argument(f'--{prefix}lr', dest=f'{prefix}lr', type=float, default=5e-4)
        parser.add_argument(f'--{prefix}weight_decay', dest=f'{prefix}weight_decay', type=float, default=1e-4)
        
        return parent_parser
