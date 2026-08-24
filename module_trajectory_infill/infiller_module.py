# Copyright (c) 2026, Da Saem Lee. All rights reserved.
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
from typing import Mapping

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors, rotation_matrix, weight_init, wrap_angle
from layers import sinusoidal_embedding

class Infiller(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 init_timestep: int,
                 num_infill_steps: int,
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        
        super().__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_timestep = init_timestep
        self.num_infill_steps = num_infill_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3


        self.mode_emb = nn.Embedding(1, hidden_dim)

        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_s2f_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)

        self.pl2m_propose_delta_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_delta_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )


        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.mode = nn.Embedding(4, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.project_propose = MLPLayer(input_dim=output_dim * self.num_infill_steps, hidden_dim=hidden_dim,
                                           output_dim= hidden_dim)
        self.to_propose_pos_delta = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim= output_dim)
        
        self.to_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim= output_dim * self.num_infill_steps)

        self.bridge_proj = MLPLayer(input_dim=self.num_infill_steps * input_dim, hidden_dim=hidden_dim,
                                           output_dim= hidden_dim)
        self.seq_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.type_a_emb = nn.Embedding(10, hidden_dim)

        
        self.apply(weight_init)

    def _relation_inputs(self, data, start_position, start_heading):
        """Build map-to-agent and agent-to-agent graph inputs."""
        agent_mask = data['agent']['mask']
        agent_batch = data['agent']['batch'][agent_mask] if isinstance(data, Batch) else None
        heading_vector = torch.stack(
            (start_heading.cos(), start_heading.sin()), dim=-1)

        map_position = data['map_polygon']['position'][:, :self.input_dim]
        map_heading = data['map_polygon']['orientation']
        map_edges = radius(
            x=start_position[:, :2],
            y=map_position[:, :2],
            r=self.pl2m_radius,
            batch_x=agent_batch,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300,
        )
        map_delta = map_position[map_edges[0]] - start_position[map_edges[1]]
        map_heading_delta = wrap_angle(
            map_heading[map_edges[0]] - start_heading[map_edges[1]])
        map_relation = torch.stack((
            torch.norm(map_delta[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=heading_vector[map_edges[1]], nbr_vector=map_delta[:, :2]),
            map_heading_delta,
        ), dim=-1)
        map_relation = self.r_pl2m_emb(
            continuous_inputs=map_relation, categorical_embs=None)

        agent_edges = radius_graph(
            x=start_position[:, :2],
            r=self.a2m_radius,
            batch=agent_batch,
            loop=False,
            max_num_neighbors=300,
        )
        agent_delta = start_position[agent_edges[0]] - start_position[agent_edges[1]]
        agent_heading_delta = wrap_angle(
            start_heading[agent_edges[0]] - start_heading[agent_edges[1]])
        agent_relation = torch.stack((
            torch.norm(agent_delta[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=heading_vector[agent_edges[1]], nbr_vector=agent_delta[:, :2]),
            agent_heading_delta,
        ), dim=-1)
        agent_relation = self.r_a2m_emb(
            continuous_inputs=agent_relation, categorical_embs=None)
        return map_relation, map_edges, agent_relation, agent_edges

    def _mode_coefficients(self, time, device):
        coefficients = torch.ones(
            4, 1, self.num_infill_steps, 1, device=device)
        coefficients[0] = time * (1 - time)
        coefficients[1] = time
        coefficients[2] = 1 - time
        return coefficients

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                gt_init,
                gt_final,
                mode) -> torch.Tensor:
        start_position = gt_init[..., :self.input_dim]
        final_position = gt_final[..., :self.input_dim]
        start_heading = gt_init[..., 2]
        x_pl = scene_enc['x_pl']
        r_pl2m, edge_index_pl2m, r_a2m, edge_index_a2m = self._relation_inputs(
            data, start_position, start_heading)

        m = self.mode_emb.weight.repeat(start_position.shape[0], 1)
        agent_types = data['agent']['type'][data['agent']['mask']].long()
        m = m + self.type_a_emb(agent_types)

        time = torch.linspace(
            0, 1, self.num_infill_steps, device=m.device).view(
                1, self.num_infill_steps, 1)
        bridge = time * (final_position - start_position).unsqueeze(1)
        bridge = torch.bmm(bridge, rotation_matrix(start_heading))
        bridge_emb = self.bridge_proj(
            bridge.reshape(-1, self.num_infill_steps * self.input_dim))
        mode_emb = self.mode(mode)
        for i in range(self.num_layers):
            m = m + bridge_emb
            m = self.pl2m_propose_delta_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_propose_delta_attn_layers[i](m, r_a2m, edge_index_a2m)

        m = m.reshape(-1, 1, self.hidden_dim)
        timestep_emb = sinusoidal_embedding(self.num_infill_steps, self.hidden_dim).to(m.device)
        m = m + timestep_emb.unsqueeze(0)
        m = self.seq_encoder(m)

        coeff = self._mode_coefficients(time, m.device)
        delta_propose = self.to_propose_pos_delta(m).view(-1, self.num_infill_steps, self.output_dim)
        delta_propose = coeff[mode].squeeze(1) * delta_propose
        loc_propose_coarse = delta_propose + bridge

        m = self.project_propose(
            loc_propose_coarse.reshape(-1, self.num_infill_steps * self.output_dim))
        for i in range(self.num_layers):
            m = m + mode_emb
            m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_propose_attn_layers[i](m, r_a2m, edge_index_a2m)

        loc_propose_pos = self.to_propose_pos(m).view(
            -1, 1, self.num_infill_steps, self.output_dim)
        loc_propose_pos = coeff[mode] * loc_propose_pos
        return loc_propose_coarse.unsqueeze(1) + loc_propose_pos
