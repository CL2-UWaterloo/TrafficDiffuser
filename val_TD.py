from argparse import ArgumentParser, SUPPRESS
import warnings

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from pipeline.infilling_pipeline import TrajInfill
from pipeline.hl_generation_pipeline import HLTrafficGen
from pipeline.end_to_end_pipeline import End2EndPipeline
from transforms import TargetBuilder
from utils import parse_bool

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

MODEL_SPECS = {
    'hl': (HLTrafficGen, 'hl_ckpt_path'),
    'infill': (TrajInfill, 'infill_ckpt_path'),
}


def model_args(args, mode, prefixed):
    """Return common arguments plus de-prefixed arguments for one model."""
    if not prefixed:
        return args
    values = {key: value for key, value in vars(args).items()
              if not key.startswith(('hl_', 'infill_'))}
    prefix = f'{mode}_'
    values.update({key.removeprefix(prefix): value
                   for key, value in vars(args).items() if key.startswith(prefix)})
    return type(args)(**values)


def load_models(args):
    """Load a single task, two independent tasks, or the chained pipeline."""
    if args.mode == 'end2end':
        runtime_args = model_args(args, 'hl', prefixed=True)
        # model_args de-prefixes HL options, but add_extra_param uses the
        # explicit checkpoint names to label validation outputs.
        runtime_args.hl_ckpt_path = args.hl_ckpt_path
        runtime_args.infill_ckpt_path = args.infill_ckpt_path
        return [('end2end', End2EndPipeline.from_checkpoints(
            runtime_args, args.hl_ckpt_path, args.infill_ckpt_path))]

    modes = ('hl', 'infill') if args.mode == 'both' else (args.mode,)
    loaded = []
    for mode in modes:
        model_class, checkpoint_arg = MODEL_SPECS[mode]
        runtime_args = model_args(args, mode, prefixed=len(modes) == 2)
        model = model_class.load_from_checkpoint(
            checkpoint_path=getattr(args, checkpoint_arg), map_location='cpu',
            strict=False, weights_only=False)
        model.add_extra_param(runtime_args)
        if mode == 'hl':
            model.check_param()
        loaded.append((mode, model))
    return loaded


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--plot', type=parse_bool, default=True)
    parser.add_argument('--network_mode', choices=['val', 'test'], default='val')
    parser.add_argument('--mode', choices=['hl', 'infill', 'both', 'end2end'], default='hl',
                        help="Validate one model, both independently, or the chained end-to-end pipeline.")

    partial_args, _ = parser.parse_known_args()
    if partial_args.mode == 'hl':
        parser.add_argument('--hl_ckpt_path', type=str, required=True)
        HLTrafficGen.add_model_specific_args(parser, training=False, prefix='')
    elif partial_args.mode == 'infill':
        parser.add_argument('--infill_ckpt_path', type=str, required=True)
        TrajInfill.add_model_specific_args(parser, training=False, prefix='')
        # Keep old validation commands usable. These options only affect the HL
        # diffusion model, so they are accepted and ignored for infill-only runs.
        parser.add_argument('--sampling', choices=['ddpm', 'ddim'], help=SUPPRESS)
        parser.add_argument('--sampling_stride', type=int, help=SUPPRESS)
    else:
        parser.add_argument('--hl_ckpt_path', type=str, required=True)
        parser.add_argument('--infill_ckpt_path', type=str, required=True)
        HLTrafficGen.add_model_specific_args(parser, training=False, prefix='hl_')
        TrajInfill.add_model_specific_args(parser, training=False, prefix='infill_')
        # Unprefixed spellings are convenient aliases for the HL model.
        parser.add_argument('--sampling', dest='hl_sampling', choices=['ddpm', 'ddim'], default=SUPPRESS)
        parser.add_argument('--sampling_stride', dest='hl_sampling_stride', type=int, default=SUPPRESS)
        parser.add_argument('--save_diffusion_steps', dest='hl_save_diffusion_steps',
                            action='store_true', default=SUPPRESS)
        parser.add_argument('--guid_task', dest='hl_guid_task',
                            choices=['none', 'map', 'map_collision', 'original'], default=SUPPRESS)
        parser.add_argument('--cost_param_costl', dest='hl_cost_param_costl',
                            type=float, default=SUPPRESS)
        parser.add_argument('--cost_param_threl', dest='hl_cost_param_threl',
                            type=float, default=SUPPRESS)
    return parser


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)
    args = build_parser().parse_args()
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='auto')

    for mode, model in load_models(args):
        num_infill_steps = model.num_infill_steps if mode in ('infill', 'end2end') else -1
        dataset = {'argoverse_v2': ArgoverseV2Dataset}[model.dataset](
            root=args.root, split=args.network_mode,
            transform=TargetBuilder(model.init_timestep, num_infill_steps))
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers and args.num_workers > 0)

        print(f'Running {args.network_mode} for {mode} model')
        if args.network_mode == 'val':
            trainer.validate(model, dataloader)
        else:
            trainer.test(model, dataloader)
