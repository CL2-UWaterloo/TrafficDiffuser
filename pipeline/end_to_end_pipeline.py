import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch_geometric.data import Batch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

from pipeline.hl_generation_pipeline import HLTrafficGen
from pipeline.infilling_pipeline import TrajInfill
from visualization import visualize_hl_scenario_prediction, visualize_scenario_infilling_prediction


class End2EndPipeline(pl.LightningModule):
    """Run high-level endpoint generation followed by trajectory infilling."""

    def __init__(self, hl_model, infill_model):
        super().__init__()
        self.hl_model = hl_model
        self.infill_model = infill_model
        self.dataset = infill_model.dataset
        self.init_timestep = infill_model.init_timestep
        self.num_infill_steps = infill_model.num_infill_steps

        if hl_model.dataset != infill_model.dataset:
            raise ValueError('HL and infill checkpoints were trained on different datasets')
        if hl_model.init_timestep != infill_model.init_timestep:
            raise ValueError('HL and infill checkpoints use different init_timestep values')

    @classmethod
    def from_checkpoints(cls, args, hl_ckpt_path, infill_ckpt_path):
        hl_model = HLTrafficGen.load_from_checkpoint(
            hl_ckpt_path, map_location='cpu', strict=False, weights_only=False)
        infill_model = TrajInfill.load_from_checkpoint(
            infill_ckpt_path, map_location='cpu', strict=False, weights_only=False)
        hl_model.add_extra_param(args)
        infill_model.add_extra_param(args)
        hl_model.check_param()
        hl_exp_name = hl_ckpt_path.split('/')[-5]
        infill_exp_name = infill_ckpt_path.split('/')[-5]
        model = cls(hl_model, infill_model)
        model.plot = args.plot
        model.root = args.root
        model.data_split = args.network_mode

        model.save_dir = f'end2end_hl_{hl_exp_name}_infill_{infill_exp_name}'
        if hl_model.guid_task != 'none':
            model.save_dir += f'_guid_{hl_model.guid_task}'
        return model

    def forward(self, data):
        eval_mask = data['agent']['mask']
        init_state, final_state = self.hl_model.generate_hl_scenarios(data, eval_mask)
        trajectory = self.infill_model.infill_trajectory(data, init_state, final_state)
        return trajectory, init_state, final_state

    def _visualize(self, data, batch_idx, eval_mask, gt, prediction_world):
        gt = gt.detach().cpu().numpy()
        prediction = prediction_world.detach().cpu().numpy()
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
                'gt': gt[selected],
                'infilled': prediction[selected],
            }
            traj_visible = {
                'gt': False,
                'infilled': True,
            }
            visualize_scenario_infilling_prediction(
                scenario, static_map, additional_traj, traj_visible, save_path, data, e2e=True)

            if self.hl_model.save_diffusion_steps:
                gt_endpoints = gt[selected][:, [0, -1], :2]
                for step, endpoints in enumerate(self.hl_model.intermediate_endpoints):
                    hl_traj = {
                        'gt': gt_endpoints,
                        'gen_hl': endpoints[selected, None].numpy(),
                    }
                    hl_visible = {'gt': False, 'gen_hl': True}
                    step_path = output_dir / (
                        f'r{rank}_b{batch_idx}_s{scene_i}_diffusion_{step:04d}.svg')
                    visualize_hl_scenario_prediction(
                        scenario, static_map, hl_traj, hl_visible, step_path, data)

    def validation_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        trajectory, init_state, _ = self(data)
        eval_mask = data['agent']['mask']
        gt = data['agent']['position'][eval_mask, self.init_timestep:]

        prediction = trajectory[..., :self.infill_model.output_dim]
        prediction_world = self.infill_model.trajectory_to_world(prediction, init_state)

        if self.plot or self.hl_model.save_diffusion_steps:
            self._visualize(data, batch_idx, eval_mask, gt, prediction_world)

    def test_step(self, data, batch_idx):
        return self.validation_step(data, batch_idx)

    def configure_optimizers(self):
        return None
