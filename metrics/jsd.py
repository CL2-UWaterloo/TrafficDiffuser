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
from pathlib import Path

import torch
from torchmetrics import Metric
from torch_cluster import radius, radius_graph

from scipy.spatial.distance import jensenshannon


class JSD_LOCAL_DENSITY(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_LOCAL_DENSITY, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.num_bin = 150
        self.local_density_pred = torch.zeros(self.num_bin)
        self.local_density_gt = torch.zeros(self.num_bin)
        


    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
                data) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        if self.local_density_pred.device != pred.device:
            self.local_density_pred = self.local_density_pred.to(pred.device)
            self.local_density_gt = self.local_density_gt.to(pred.device)

        edge_pred = radius_graph(pred, r = torch.inf, batch = agent_batch, max_num_neighbors=agent_batch.bincount().max())
        rel_pos = pred[edge_pred[1]] - pred[edge_pred[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        top5_dist_list = []
        for agent_i in range(pred.size(0)):
            agent_idx = torch.where(edge_pred[1] == agent_i)[0]
            if agent_idx.size(0) == 0:
                continue
            min_dist = torch.topk(dist[agent_idx], min(5, len(agent_idx)), largest=False)
            top5_dist_list.append(min_dist.values)
        top5_dist_list = torch.cat(top5_dist_list)
        self.local_density_pred += torch.histc(top5_dist_list, bins=self.num_bin, min=0, max=self.num_bin)


        edge_gt = radius_graph(gt, r = torch.inf, batch = agent_batch, max_num_neighbors=agent_batch.bincount().max())
        rel_pos = gt[edge_gt[1]] - gt[edge_gt[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        top5_dist_list = []
        for agent_i in range(pred.size(0)):
            agent_idx = torch.where(edge_pred[1] == agent_i)[0]
            if agent_idx.size(0) == 0:
                continue
            min_dist = torch.topk(dist[agent_idx], min(5, len(agent_idx)), largest=False)
            top5_dist_list.append(min_dist.values)
        top5_dist_list = torch.cat(top5_dist_list)

        self.local_density_gt += torch.histc(top5_dist_list, bins=self.num_bin, min=0, max=self.num_bin)


    def compute(self) -> torch.Tensor:
        
        dist_cumulative_pred = self.local_density_pred / self.local_density_pred.sum() 
        dist_cumulative_gt = self.local_density_gt / self.local_density_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        
        return torch.Tensor([jsd_dist]).to(self.local_density_gt.device)


class JSD_MAP_DIST(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_MAP_DIST, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

        self.hist_dist_a2map_pred = torch.zeros(30)
        self.hist_dist_a2map_gt = torch.zeros(30)


    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
                map_pts) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        
        if  self.hist_dist_a2map_pred.device != pred.device:
            self.hist_dist_a2map_pred = self.hist_dist_a2map_pred.to(pred.device)
            self.hist_dist_a2map_gt = self.hist_dist_a2map_gt.to(pred.device)

        center_mask = map_pts['side'] == 2
        map_pt_batch = map_pts['batch'][center_mask]
        max_neighbor = map_pt_batch.bincount().max().item()
        map_pts_pos = map_pts['position'][center_mask, :2]
        
        min_dist_a2edge_pred = torch.zeros_like(pred[:, 0])
        pred_pos = pred[:, :2]
        edge_a2m = radius(pred_pos, map_pts_pos, r = torch.inf, batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        
        rel_pos = pred_pos[edge_a2m[1]] - map_pts_pos[edge_a2m[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        for batch_i in range(pred_pos.size(0)):
            agent_i = (edge_a2m[1] == batch_i)
            if agent_i.sum() == 0:
                continue
            agent_idx = torch.where(agent_i)[0]
            min_dist_idx = agent_idx[dist[agent_i].argmin()]
            self.closest_map_idx_pred = min_dist_idx
            min_dist_a2edge_pred[batch_i] = dist[min_dist_idx]

        self.hist_dist_a2map_pred += torch.histc(min_dist_a2edge_pred, bins=30, min=0, max=3)
        
        min_dist_a2edge_gt = torch.zeros_like(pred[:, 0])
        gt_pos = gt[:, :2]
        edge_a2m = radius(gt_pos, map_pts_pos, r = torch.inf, batch_x = agent_batch, batch_y = map_pt_batch, max_num_neighbors=max_neighbor)
        
        rel_pos = gt_pos[edge_a2m[1]] - map_pts_pos[edge_a2m[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)
        for batch_i in range(gt_pos.size(0)):
            agent_i = (edge_a2m[1] == batch_i)
            if agent_i.sum() == 0:
                continue
            agent_idx = torch.where(agent_i)[0]
            min_dist_idx = agent_idx[dist[agent_i].argmin()]
            self.closest_map_idx_gt = min_dist_idx
            min_dist_a2edge_gt[batch_i] = dist[min_dist_idx]

        self.hist_dist_a2map_gt += torch.histc(min_dist_a2edge_gt, bins=30, min=0, max=3)
        
    def compute(self) -> torch.Tensor:
        dist_cumulative_pred = self.hist_dist_a2map_pred / self.hist_dist_a2map_pred.sum()
        dist_cumulative_gt = self.hist_dist_a2map_gt / self.hist_dist_a2map_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        
        return torch.Tensor([jsd_dist]).to(dist_cumulative_gt.device)


class JSD_INTERACTIVE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_INTERACTIVE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        
        self.hist_dist_a2a_pred = torch.zeros(250)
        self.hist_dist_a2a_gt = torch.zeros(250)

    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
               agent_batch: torch.Tensor,
               ) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)

        edge_a2a = radius_graph(pred, r=torch.inf, batch=agent_batch, max_num_neighbors=agent_batch.bincount().max().item())
        rel_pos = pred[edge_a2a[1]] - pred[edge_a2a[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)

        min_dist_a2a_pred = torch.zeros_like(pred[:, 0])
        
        for batch_i in range(pred.size(0)):
            agent_i = (edge_a2a[1] == batch_i) | (edge_a2a[0] == batch_i)
            if agent_i.sum() == 0:
                continue

            min_dist_a2a_pred[batch_i] = dist[agent_i].min()

        if self.hist_dist_a2a_pred.device != min_dist_a2a_pred.device:
            self.hist_dist_a2a_pred = self.hist_dist_a2a_pred.to(min_dist_a2a_pred.device)
        self.hist_dist_a2a_pred += torch.histc(min_dist_a2a_pred, bins=250, min=0, max=250)


        edge_a2a = radius_graph(gt, r=torch.inf, batch=agent_batch, max_num_neighbors=agent_batch.bincount().max().item())
        rel_pos = gt[edge_a2a[1]] - gt[edge_a2a[0]]
        dist = torch.norm(rel_pos[:, :2], p=2, dim=-1)

        
        min_dist_a2a_gt = torch.zeros_like(gt[:, 0])
        for batch_i in range(gt.size(0)):
            agent_i = (edge_a2a[1] == batch_i) | (edge_a2a[0] == batch_i)
            if agent_i.sum() == 0:
                continue

            min_dist_a2a_gt[batch_i] = dist[agent_i].min()

        if self.hist_dist_a2a_gt.device != min_dist_a2a_gt.device:
            self.hist_dist_a2a_gt = self.hist_dist_a2a_gt.to(min_dist_a2a_gt.device)
        self.hist_dist_a2a_gt += torch.histc(min_dist_a2a_gt, bins=250, min=0, max=250)


    def compute(self) -> torch.Tensor:
        dist_cumulative_pred = self.hist_dist_a2a_pred / self.hist_dist_a2a_pred.sum()
        dist_cumulative_gt = self.hist_dist_a2a_gt / self.hist_dist_a2a_gt.sum()
        jsd_dist = jensenshannon(dist_cumulative_pred.detach().cpu().numpy(), dist_cumulative_gt.detach().cpu().numpy())
        return torch.Tensor([jsd_dist]).to(self.hist_dist_a2a_pred.device)

class JSD_SPEED(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(JSD_SPEED, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        # ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
        #                      'construction', 'riderless_bicycle', 'unknown']

        self.hist_speed_pred = torch.zeros(50)
        self.hist_speed_gt = torch.zeros(50)

    def update(self,
               pred: torch.Tensor,
               gt: torch.Tensor,
                ) -> None:
        if len(pred.shape) == 3:
            pred = pred.squeeze(1)
        # pred = (pred + 1) * 15
        # gt = (gt + 1) * 15
        if self.hist_speed_pred.device != pred.device:
            self.hist_speed_pred = self.hist_speed_pred.to(pred.device)
            self.hist_speed_gt = self.hist_speed_gt.to(pred.device)
        self.hist_speed_pred += torch.histc(pred, bins=50, min=0, max=50)
        self.hist_speed_gt += torch.histc(gt, bins=50, min=0, max=50)
        
            
    def compute(self) -> torch.Tensor:
        speed_cumulative_pred = self.hist_speed_pred / self.hist_speed_pred.sum()
        speed_cumulative_gt = self.hist_speed_gt / self.hist_speed_gt.sum()
        jsd_speed = jensenshannon(speed_cumulative_pred.detach().cpu().numpy(), speed_cumulative_gt.detach().cpu().numpy())
        return torch.Tensor([jsd_speed]).to(self.hist_speed_pred.device)