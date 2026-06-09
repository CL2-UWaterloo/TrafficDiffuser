# TrafficDiffuser

This repository is the official github repository for [TrafficDiffuser].

 [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://dasaemlee.github.io/projects/trafficdiffuser/) 
 <!-- [![arXiv](https://img.shields.io/badge/arXiv-COVE-b31b1b.svg)](https://arxiv.org/abs/2406.08850)  -->

> **Top-down Traffic Scenario Generation via Joint Initial-Goal Diffusion and Trajectory Infilling**  
> [Da Saem Lee](https://dasaemlee.github.io),
> [Yash Vardhan Pant](https://yashpant.github.io),
> [Sebastian Fischmeister](https://uwaterloo.ca/embedded-software-group/people-profiles/sebastian-fischmeister),



<p>
Robust traffic simulators are crucial for developing and testing autonomous vehicles to reduce the costly, labor-intensive real-world data collection process and the need for physical presence on the road. However, existing simulators require agents' initial states to generate trajectories, which limits scalability and diversity due to restrictions on the given initial states. While data-driven agent initialization has been widely studied, the generated initial states are not interpretable in terms of why the agents are initialized at those specific locations. Given known initial states, trajectory generation is also a challenging problem, as the model must learn the variability of the destination and how agents should reach it over time. In this paper, we propose TrafficDiffuser, a top-down traffic scenario generation framework that generates high-level traffic scenarios, defined by initial and goal state pairs, by jointly modeling them. The high-level scenario generation makes initial states better interpretable and reduces trajectory generation into as simple as an infilling problem. We demonstrate how the generated high-level traffic scenarios can be used, including constraining based on different trajectory modes and integrating them with existing trajectory generation models. We conduct extensive experiments on the Argoverse 2 motion prediction dataset to evaluate how well the generated outputs capture real-world distributions.  In addition to generating goal states, TrafficDiffuser outperforms the next-best approach for agent initialization, reducing speed distribution distance by 55.3% and the off-road rate by 2.8%. 
</p>

## News
- [2026.5.1] Paper is accepted by [ITSC 2026](https://ieee-itsc.org/2026/)

## Codes

Coming soon!

## Visualization
For more visualizations, please visit the project website:
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://dasaemlee.github.io/projects/trafficdiffuser/)

