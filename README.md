# TrafficDiffuser

This repository is the official github repository for TrafficDiffuser.

 [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://dasaemlee.github.io/projects/trafficdiffuser/) 
 [![arXiv](https://img.shields.io/badge/arXiv-COVE-b31b1b.svg)](https://arxiv.org/abs/2608.11407) 

> **Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling**  
> [Da Saem Lee](https://dasaemlee.github.io),
> [Yash Vardhan Pant](https://yashpant.github.io),
> [Sebastian Fischmeister](https://uwaterloo.ca/embedded-software-group/people-profiles/sebastian-fischmeister),


<p>
Robust traffic simulators are crucial for developing and testing autonomous vehicles to reduce the costly, labor-intensive real-world data collection process and the need for physical presence on the road. However, existing simulators require agents' initial states to generate trajectories, which limits scalability and diversity due to restrictions on the given initial states. While data-driven agent initialization has been widely studied, the generated initial states are not interpretable in terms of why the agents are initialized at those specific locations. Given known initial states, trajectory generation is also a challenging problem, as the model must learn the variability of the destination and how agents should reach it over time. In this paper, we propose TrafficDiffuser, a top-down traffic scenario generation framework that generates high-level traffic scenarios, defined by initial and goal state pairs, by jointly modeling them. The high-level scenario generation makes initial states better interpretable and reduces trajectory generation into as simple as an infilling problem. We demonstrate how the generated high-level traffic scenarios can be used, including constraining based on different trajectory modes and integrating them with existing trajectory generation models. We conduct extensive experiments on the Argoverse 2 motion prediction dataset to evaluate how well the generated outputs capture real-world distributions.  In addition to generating goal states, TrafficDiffuser outperforms the next-best approach for agent initialization, reducing speed distribution distance by 55.3% and the off-road rate by 2.8%. 
</p>

## News
- [2026.5.1] Paper is accepted by [ITSC 2026](https://ieee-itsc.org/2026/)


## Set up

**Step 1**: Clone the repository and create the environment:

```bash
git clone git@github.com:CL2-UWaterloo/TrafficDiffuser.git && cd TrafficDiffuser
uv sync --locked
```
This reads the dependency list from `pyproject.toml` and installs the project into a local `.venv` managed by `uv`.


**Step 2**: Download the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html) and make sure the dataset root is available on your machine. The project expects the dataset path via `--root <dataset-root>` in the training/validation commands below.

For the dataset client/API setup, follow the official [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html) and the [Argoverse 2 API](https://github.com/argoverse/av2-api).

**Step 3**: Activate the environment:
```bash
source .venv/bin/activate
```

## Training

The two components are trained separately. Both commands use the same entry point and select the component with `--mode`.

### High-level scenario model

```sh
python train_TD.py \
  --mode hl \
  --root <dataset-root> \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --init_timestep 50 \
  --pl2pl_radius 150 \
  --num_denoiser_layers 3 \
  --num_diffusion_steps 500 \
  --T_max 30 \
  --lr 0.001 \
  --beta_1 0.0001 \
  --beta_T 0.05 \
  --sampling ddim \
  --sampling_stride 10 \
  --devices "-1" \
  --num_workers 16 \
  --max_epochs 30
```

### Trajectory infilling model

```sh
python train_TD.py \
  --mode infill \
  --root <dataset-root> \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --init_timestep 50 \
  --num_infill_steps 60 \
  --pl2pl_radius 150 \
  --a2a_radius 50 \
  --pl2m_radius 150 \
  --a2m_radius 150 \
  --lr 0.0001 \
  --devices "-1" \
  --num_workers 16 \
  --max_epochs 30
```

Training logs and checkpoints are written under `logs_hl/` or `logs_infill/`. Set `--plot true` only when training-time visualization is needed. It defaults to `false`.

## Validation

Plotting accepts Boolean values through `--plot true` and `--plot false`.

### Validate the high-level model

```sh
python val_TD.py \
  --mode hl \
  --root <dataset-root> \
  --hl_ckpt_path <hl-checkpoint> \
  --batch_size 8 \
  --devices "-1" \
  --sampling ddim \
  --sampling_stride 10 \
  --network_mode val \
  --plot false
```

To use guidance, set a task other than `none` and change the cost parameters:

```sh
  --guid_task map_collision \
  --cost_param_costl 1.0 \
  --cost_param_threl 1.0
```

Available guidance tasks are `none`, `map`, `map_collision`, and `original`.

To save the high-level state at every sampled reverse-diffusion step, add:

```sh
  --save_diffusion_steps
```

Intermediate step visualizations are saved even when `--plot false`.

### Validate the infilling model

```sh
python val_TD.py \
  --mode infill \
  --root <dataset-root> \
  --infill_ckpt_path <infill-checkpoint> \
  --batch_size 8 \
  --devices "-1" \
  --network_mode val \
  --plot false
```

### Validate the chained pipeline

End-to-end validation runs the high-level model first for each batch and passes its generated endpoint states to the infilling model.

```sh
python val_TD.py \
  --mode end2end \
  --root <dataset-root> \
  --hl_ckpt_path <hl-checkpoint> \
  --infill_ckpt_path <infill-checkpoint> \
  --batch_size 8 \
  --devices "-1" \
  --sampling ddim \
  --sampling_stride 10 \
  --guid_task none \
  --network_mode val \
  --plot true
```

Guidance and intermediate diffusion output can also be enabled for end-to-end validation:

```sh
  --guid_task map_collision \
  --save_diffusion_steps
```

Use `--mode both` with both checkpoint paths to validate the two models independently instead of chaining them.

## Visualization
For more visualizations, please visit the project website:
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://dasaemlee.github.io/projects/trafficdiffuser/)


## Citation
```
@misc{lee2026topdowntrafficscenariogeneration,
      title={Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling}, 
      author={Da Saem Lee and Yash Vardhan Pant and Sebastian Fischmeister},
      year={2026},
      eprint={2608.11407},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.11407}, 
}
```
## Acknowledgement
This code is based on [Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](https://github.com/YixiaoWang7/OptTrajDiff), [Query-Centric Trajectory Prediction](https://github.com/ZikangZhou/QCNet), and [Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator](https://github.com/CL2-UWaterloo/PathDiffuser)
Please also consider citing:

```@INPROCEEDINGS{lee2025pathdiffuserdiffusionmodel,
  author={Lee, Da Saem and Karthikeyan, Akash and Pant, Yash Vardhan and Fischmeister, Sebastian},
  booktitle={2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC)}, 
  title={Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator}, 
  year={2025},
  volume={},
  number={},
  pages={569-576},
  keywords={Training;Measurement;Dimensionality reduction;Weather;Adaptation models;Roads;Perturbation methods;Diffusion models;Controllability;Trajectory},
  doi={10.1109/ITSC60802.2025.11423013}}
```
```
@inproceedings{wang2025optimizing,
  title={Optimizing diffusion models for joint trajectory prediction and controllable generation},
  author={Wang, Yixiao and Tang, Chen and Sun, Lingfeng and Rossi, Simone and Xie, Yichen and Peng, Chensheng and Hannagan, Thomas and Sabatini, Stefano and Poerio, Nicola and Tomizuka, Masayoshi and others},
  booktitle={European Conference on Computer Vision},
  pages={324--341},
  year={2025},
  organization={Springer}
}
```
```
@inproceedings{zhou2023query,
  title={Query-Centric Trajectory Prediction},
  author={Zhou, Zikang and Wang, Jianping and Li, Yung-Hui and Huang, Yu-Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
