# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils for Argoverse MF scenarios."""

import io
import math
from pathlib import Path
from typing import Final, List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from PIL.Image import Image

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#E0E0E0"
_LANE_SEGMENT_COLOR: Final[str] = "#7A7A7A"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[
    int
] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

def visualize_scenario_infilling_prediction(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    additional_traj: dict,
    traj_visible: dict,
    save_path: Path,
    data,
    e2e=False
) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Note: This function uses OpenCV to create a MP4 file using the MP4V codec.

    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        save_path: Path where output MP4 video should be saved.
    """        
    # Build each frame for the video
    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    _, ax = plt.subplots(figsize = (20,20))

    # Plot static map elements and actor tracks
    _plot_static_map_elements_prediction(scenario_static_map)
    cur_plot_bounds = _plot_actor_tracks_prediction(ax, scenario, _OBS_DURATION_TIMESTEPS)
    plot_bounds = [1,1,1,1]
    if cur_plot_bounds:
        plot_bounds[0] = cur_plot_bounds[0]
        plot_bounds[1] = cur_plot_bounds[1]
        plot_bounds[2] = cur_plot_bounds[2]
        plot_bounds[3] = cur_plot_bounds[3]

    
    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


    
    if traj_visible['gt']:
        gt = additional_traj['gt']
        for k in range(gt.shape[0]):
            plt.plot(gt[k,:,0],gt[k,:,1],color = 'mediumseagreen',linewidth = 10,zorder = 1000, label='gt')
            plt.plot(gt[k,0,0],gt[k,0,1],marker='o', color='mediumseagreen',markersize=20,  zorder = 10000, label='Groundtruth initial point')
            plt.plot(gt[k,:,0],gt[k,:,1],color = 'mediumseagreen',linewidth = 6,alpha = 1.0, zorder = 10000, label='Groundtruth Trajectory')
            plt.plot(gt[k,0,0],gt[k,0,1],color = 'mediumseagreen', marker='o',markersize=10, label='Groundtruth Start Point', zorder = 10000)    
            i = k
            dx = gt[i,-1,0] - gt[i,-2,0]
            dy = gt[i,-1,1] - gt[i,-2,1]
            plt.arrow(gt[i,-2,0] , gt[i,-2,1], dx, dy, head_width=1.5, head_length=1.5, fc='mediumseagreen', ec='mediumseagreen', zorder = 10000)


    if traj_visible['infilled']:
        infilled = additional_traj['infilled']
        if e2e:
            legend_prefix = 'Generated'
        else:
            legend_prefix = 'Groundtruth'
        for k in range(infilled.shape[0]):
            plt.plot(infilled[k,0,:,0],infilled[k,0,:,1],color ='green' ,linewidth = 5, zorder = 10000, label='Infilled trajectory')
            
            plt.plot(infilled[k,0,0,0],infilled[k,0,0,1],color='blue' , marker='o',markersize=20, zorder = 10000, label=f'{legend_prefix} initial position')
        for k in range(infilled.shape[0]):
            plt.plot(infilled[k,0,-1,0],infilled[k,0,-1,1],color='red' , marker='*',markersize=20, zorder = 10000, label=f'{legend_prefix} final position') 

    gt_eval_world = additional_traj['infilled']
    
    plot_bounds[0] = gt_eval_world[..., 0].min()
    plot_bounds[1] = gt_eval_world[..., 0].max()
    plot_bounds[2] = gt_eval_world[..., 1].min()
    plot_bounds[3] = gt_eval_world[..., 1].max()
    
    plt.xlim(
        plot_bounds[0] - 20,
        plot_bounds[1] + 15,
    )
    plt.ylim(
        plot_bounds[2] - 30,
        plot_bounds[3] + 5,
    )
    plt.savefig(save_path, format='svg')
    plt.close()
    

def visualize_hl_scenario_prediction(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    additional_traj: dict,
    traj_visible: dict,
    save_path: Path,
    data
) -> None:
    
    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    _, ax = plt.subplots(figsize = (20,20))

    # Plot static map elements and actor tracks
    _plot_static_map_elements_prediction(scenario_static_map)
    cur_plot_bounds = _plot_actor_tracks_prediction(ax, scenario, _OBS_DURATION_TIMESTEPS)
    plot_bounds = [1,1,1,1]
    if cur_plot_bounds:
        plot_bounds[0] = cur_plot_bounds[0]
        plot_bounds[1] = cur_plot_bounds[1]
        plot_bounds[2] = cur_plot_bounds[2]
        plot_bounds[3] = cur_plot_bounds[3]

    
    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


    if traj_visible['gt']:
        gt = additional_traj['gt']
        for k in range(gt.shape[0]):
            plt.plot(gt[k, 0, 0],gt[k, 0, 1],marker='o', color='mediumseagreen',markersize=20,  zorder = 10000, label='Groundtruth initial point')
            plt.plot(gt[k,-1, 0],gt[k,-1, 1],marker='o', color='green',markersize=20, zorder = 10000, label='Groundtruth final point')

            x = gt[k, 0, 0].item(), gt[k, -1, 0].item()
            y = gt[k, 0, 1].item(), gt[k, -1, 1].item()

            plt.plot(x, y, color='green', linewidth=2, zorder = 10000)  # draw the line


    if traj_visible['gen_hl']:
        gen_hl = additional_traj['gen_hl']
        
        for k in range(gen_hl.shape[0]):            
            plt.plot(gen_hl[k, 0, 0, 0],gen_hl[k, 0, 0, 1],color='b' , marker='o',markersize=20, zorder = 10000, label='Generated initial position')
            plt.plot(gen_hl[k, 0, -1, 0],gen_hl[k, 0, -1, 1],color='r' , marker='*',markersize=20, zorder = 10000, label='Generated final position') 

            x = [gen_hl[k, 0, 0, 0].item(), gen_hl[k, 0, -1, 0].item()]
            y =[gen_hl[k, 0, 0, 1].item(), gen_hl[k, 0, -1, 1].item()]

            plt.plot(x, y, color='dodgerblue', linewidth=5, label='Pair Indicator', linestyle='-.', zorder = 10000)  # draw the line


    gt_eval_world = np.concatenate([additional_traj['gt'], additional_traj['gen_hl'].squeeze(1)], axis=1)

    plot_bounds[0] = gt_eval_world[..., 0].min()
    plot_bounds[1] = gt_eval_world[..., 0].max()
    plot_bounds[2] = gt_eval_world[..., 1].min()
    plot_bounds[3] = gt_eval_world[..., 1].max()

    plt.xlim(
        plot_bounds[0] - 20,
        plot_bounds[1] + 15,
    )
    plt.ylim(
        plot_bounds[2] - 30,
        plot_bounds[3] + 5,
    )

    plt.savefig(save_path, format='svg')
    plt.close()

    

def _plot_static_map_elements_prediction(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    # for drivable_area in static_map.vector_drivable_areas.values():
    #     _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=3,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def _plot_actor_tracks_prediction(
    ax: plt.Axes, scenario: ArgoverseScenario, timestep: int
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

            
        if track.category == TrackCategory.FOCAL_TRACK or track.category == TrackCategory.SCORED_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)


    return track_bounds

def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )





def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        angle=np.degrees(heading),
        
        # edgecolor = 'k',
        # linewidth = 2,
        facecolor=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)

def visualize_scenario_prediction_infilling_video(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    additional_traj: dict,
    traj_visible: dict,
    save_path: Path,
    data
) -> None:
    """Build an animated video/GIF showing trajectory evolution across T steps.

    GT shape: [N, T, 2]  — T is dim 1
    rec_traj shape: [N, num_samples, T, 2]  — T is dim 2

    Saves both a GIF and an MP4 at save_path (with appropriate extensions).
    """
    rec_traj = additional_traj['rec_init'] if traj_visible['rec_traj'] else None  # [N, S, T, 2]
    special_set = np.arange(rec_traj.shape[0])

    T = rec_traj.shape[2]

    # Compute fixed plot bounds from full trajectories so axes don't shift per frame
    x_vals = [rec_traj[..., 0].ravel()]
    y_vals = [rec_traj[..., 1].ravel()]
    if rec_traj is not None:
        x_vals.append(rec_traj[..., 0].ravel())
        y_vals.append(rec_traj[..., 1].ravel())
    x_min = np.min(np.concatenate(x_vals))
    x_max = np.max(np.concatenate(x_vals))
    y_min = np.min(np.concatenate(y_vals))
    y_max = np.max(np.concatenate(y_vals))

    frames: List[Image] = []

    for t in range(T):
        fig, ax = plt.subplots(figsize=(20, 20))

        _plot_static_map_elements_prediction(scenario_static_map)
        _plot_actor_tracks_prediction(ax, scenario, _OBS_DURATION_TIMESTEPS)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if traj_visible['gt']:
            for k in range(rec_traj.shape[0]):
                traj_so_far = rec_traj[k, :t + 1, :]  # [t+1, 2]
                if k in special_set:
                    if len(traj_so_far) > 1:
                        plt.plot(traj_so_far[:, 0], traj_so_far[:, 1],
                                 color='mediumseagreen', linewidth=10, zorder=1000, label='Groundtruth')
                    plt.plot(traj_so_far[0, 0], traj_so_far[0, 1],
                             marker='o', color='mediumseagreen', markersize=20,
                             label='Groundtruth Start Point', zorder=10000)
                    if len(traj_so_far) >= 2:
                        dx = traj_so_far[-1, 0] - traj_so_far[-2, 0]
                        dy = traj_so_far[-1, 1] - traj_so_far[-2, 1]
                        plt.arrow(traj_so_far[-2, 0], traj_so_far[-2, 1], dx, dy,
                                  head_width=1.5, head_length=1.5,
                                  fc='mediumseagreen', ec='mediumseagreen', zorder=1000)
                    # Bounding box at current position
                    if t > 0:
                        gt_heading = math.atan2(rec_traj[k, t, 1] - rec_traj[k, t - 1, 1],
                                                rec_traj[k, t, 0] - rec_traj[k, t - 1, 0])
                    else:
                        gt_heading = math.atan2(rec_traj[k, min(1, T - 1), 1] - rec_traj[k, 0, 1],
                                                rec_traj[k, min(1, T - 1), 0] - rec_traj[k, 0, 0])
                    _plot_actor_bounding_box(ax, rec_traj[k, t, :], gt_heading,
                                             'mediumseagreen',
                                             (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M))
                else:
                    if len(traj_so_far) > 1:
                        plt.plot(traj_so_far[:, 0], traj_so_far[:, 1],
                                 color='mediumseagreen', linewidth=10, zorder=1000)

        if rec_traj is not None:
            for k in range(rec_traj.shape[0]):
                if k in special_set:
                    for i in range(rec_traj.shape[1]):
                        traj_so_far = rec_traj[k, i, :t + 1, :]  # [t+1, 2]
                        if len(traj_so_far) > 1:
                            plt.plot(traj_so_far[:, 0], traj_so_far[:, 1],
                                     color='dodgerblue', linewidth=6, alpha=1.0,
                                     zorder=10000, label='Prediction')
                        # Bounding box at current position for each sample
                        if t > 0:
                            pred_heading = math.atan2(rec_traj[k, i, t, 1] - rec_traj[k, i, t - 1, 1],
                                                      rec_traj[k, i, t, 0] - rec_traj[k, i, t - 1, 0])
                        else:
                            pred_heading = math.atan2(rec_traj[k, i, min(1, T - 1), 1] - rec_traj[k, i, 0, 1],
                                                      rec_traj[k, i, min(1, T - 1), 0] - rec_traj[k, i, 0, 0])
                        _plot_actor_bounding_box(ax, rec_traj[k, i, t, :], pred_heading,
                                                 'dodgerblue',
                                                 (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M))
                    plt.plot(rec_traj[k, 0, 0, 0], rec_traj[k, 0, 0, 1],
                             color='dodgerblue', marker='o', markersize=10,
                             label='Prediction Start Point', zorder=10000)
                    if t >= 1:
                        dx = rec_traj[k, 0, t, 0] - rec_traj[k, 0, t - 1, 0]
                        dy = rec_traj[k, 0, t, 1] - rec_traj[k, 0, t - 1, 1]
                        plt.arrow(rec_traj[k, 0, t - 1, 0], rec_traj[k, 0, t - 1, 1], dx, dy,
                                  head_width=1.5, head_length=1.5,
                                  fc='dodgerblue', ec='dodgerblue', zorder=10000)
                else:
                    for i in range(rec_traj.shape[1]):
                        traj_so_far = rec_traj[k, i, :t + 1, :]
                        if len(traj_so_far) > 1:
                            plt.plot(traj_so_far[:, 0], traj_so_far[:, 1],
                                     color='dodgerblue', linewidth=6, alpha=1, zorder=1000)
        
        plot_size = max(x_max - x_min, y_max - y_min) + 30
        mid_pt = (x_max + x_min) / 2, (y_max + y_min) / 2
        plt.xlim(mid_pt[0] - plot_size / 2, mid_pt[0] + plot_size / 2)
        plt.ylim(mid_pt[1] - plot_size / 2, mid_pt[1] + plot_size / 2)

        ax.text(0.02, 0.98, f't = {t + 1} / {T}', transform=ax.transAxes,
                fontsize=24, verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    # Save as GIF
    gif_path = str(save_path).replace('.pdf', '.gif').replace('.svg', '.gif')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    # Save as MP4
    mp4_path = str(save_path).replace('.pdf', '.mp4').replace('.svg', '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_path, fourcc, fps=10, frameSize=frames[0].size)
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
    
