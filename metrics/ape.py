import typing

import numpy as np
from evo.core import lie_algebra, metrics, sync
from evo.core.metrics import PoseRelation
from evo.core.result import Result
from evo.core.trajectory import Plane, PosePath3D, PoseTrajectory3D
from scipy.interpolate import interp1d

from utils.image_utils import (
    batch_transform,
    get_act_obs_reverse_transforms,
    get_start_end_reverse_transform,
)
from utils.path_utils import convert_image_to_path


def theta_to_quaternion(theta):
    """Convert a heading (theta) to a quaternion representation"""
    qx = 0
    qy = 0
    qz = np.sin(theta / 2)
    qw = np.cos(theta / 2)
    return qx, qy, qz, qw


def convert_path_to_poses(path_xy):
    poses = []

    for i in range(len(path_xy) - 1):
        timestamp = i
        x, y = path_xy[i, 0], path_xy[i, 1]
        x_next, y_next = path_xy[i + 1, 0], path_xy[i + 1, 1]

        theta = np.arctan2(y_next - y, x_next - x)
        quaternion = theta_to_quaternion(theta)

        poses.append(
            {
                "timestamp": i,
                "pose": {
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "quaternion": quaternion,
                },
            }
        )
    return poses


def convert_poses_to_PoseTrajectory3D(poses):
    xyz = [
        np.array([pose["pose"]["x"], pose["pose"]["y"], 0]) for pose in poses
    ]
    quat = [np.array(pose["pose"]["quaternion"]) for pose in poses]
    stamps = [np.array(pose["timestamp"]) for pose in poses]
    return PoseTrajectory3D(xyz, quat, np.asarray(stamps, dtype=np.float64))


def batch_ape_from_prediction(ground_truth, pred, align=False):
    reverse_start_end_transform = get_start_end_reverse_transform(False)
    (
        act_reverse_transform,
        obs_reverse_transform,
    ) = get_act_obs_reverse_transforms(tensor=False)

    gt_path, batch_start, batch_end = ground_truth
    # Reverse normalization
    batch_start, batch_end = batch_transform(
        batch_start.cpu(), reverse_start_end_transform
    ), batch_transform(batch_end.cpu(), reverse_start_end_transform)
    gt_path = batch_transform(gt_path.cpu(), act_reverse_transform)

    # Convert to binary images
    gt_path = np.asarray(gt_path[..., 0] / 255.0, dtype=np.uint8)

    results = {
        "rmse": [],
        "mean": [],
        "median": [],
        "std": [],
        "min": [],
        "max": [],
        "sse": [],
    }

    for i in range(len(batch_start)):
        # if i <=5 :
        #     continue
        # print("Processing index: ", i)

        binary_gt, binary_pred, start, end = (
            gt_path[i],
            pred[i],
            batch_start[i],
            batch_end[i],
        )
        output = calculate_ape_from_prediction(
            binary_gt, binary_pred, start, end, align=align
        )

        if output:
            (vis_res, res, traj_est), traj_ref = output
            # callback_ape(vis_res, traj_ref, traj_est, PlotMode.xy, True)
            for key, val in res.items():
                if not np.isnan(val):
                    results[key].append(val)
    if len(results["rmse"]) < 1:
        return None
    return {key: np.mean(val) for key, val in results.items()}


def interpolate_path_data(path_data, required_timestamps):
    original_timestamps = [data["timestamp"] for data in path_data]
    xs = [data["pose"]["x"] for data in path_data]
    ys = [data["pose"]["y"] for data in path_data]
    thetas = [data["pose"]["theta"] for data in path_data]

    interpolate_x = interp1d(
        original_timestamps, xs, kind="linear", fill_value="extrapolate"
    )
    interpolate_y = interp1d(
        original_timestamps, ys, kind="linear", fill_value="extrapolate"
    )
    interpolate_theta = interp1d(
        original_timestamps, thetas, kind="linear", fill_value="extrapolate"
    )

    interpolated_data = []
    for i, t in enumerate(required_timestamps):
        x = interpolate_x(t)
        y = interpolate_y(t)
        theta = interpolate_theta(t)
        quaternion = theta_to_quaternion(theta)
        interpolated_data.append(
            {
                "timestamp": i,
                "pose": {
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "quaternion": quaternion,
                },
            }
        )

    return interpolated_data


def calculate_ape_from_prediction(
    binary_gt, binary_pred, start, end, align=False
):
    # convert binary images to path
    # No need to pass gt through clusterer, TODO fix

    gt_path = convert_image_to_path(binary_gt, start, end, pred=True)

    pred_path = convert_image_to_path(
        binary_pred[0, ...], start, end, pred=True
    )

    if pred_path is None or gt_path is None:
        return None

    if len(pred_path) <= 1 or len(gt_path) <= 1:
        return None

    # convert 2D path to 3D poses with timestamps
    gt_poses = convert_path_to_poses(gt_path)
    pred_poses = convert_path_to_poses(pred_path)

    if len(pred_poses) < len(gt_poses):
        pred_poses = interpolate_path_data(
            pred_poses,
            required_timestamps=np.linspace(
                0, len(pred_poses) - 1, num=len(gt_poses)
            ),
        )
    elif len(pred_poses) > len(gt_poses):
        gt_poses = interpolate_path_data(
            gt_poses,
            required_timestamps=np.linspace(
                0, len(gt_poses) - 1, num=len(pred_poses)
            ),
        )

    # convert custom 3D trajectory to Evo PoseTrajectory3D
    traj_ref = convert_poses_to_PoseTrajectory3D(gt_poses)
    traj_est = convert_poses_to_PoseTrajectory3D(pred_poses)

    result: Result = ape(
        traj_ref,
        traj_est,
        est_name=None,
        pose_relation=PoseRelation.translation_part,
        align=align,
    )

    return result, traj_ref


def ape(
    traj_ref: PosePath3D,
    traj_est: PosePath3D,
    pose_relation: metrics.PoseRelation,
    align: bool = False,
    correct_scale: bool = False,
    n_to_align: int = -1,
    align_origin: bool = False,
    ref_name: str = "reference",
    est_name: str = "estimate",
    change_unit: typing.Optional[metrics.Unit] = None,
    project_to_plane: typing.Optional[Plane] = None,
):
    # Align the trajectories.
    only_scale = correct_scale and not align
    alignment_transformation = None
    if align or correct_scale:
        alignment_transformation = lie_algebra.sim3(
            *traj_est.align(traj_ref, correct_scale, only_scale, n=n_to_align)
        )
    elif align_origin:
        alignment_transformation = traj_est.align_origin(traj_ref)

    # Projection is done after potential 3D alignment & transformation steps.
    if project_to_plane:
        traj_ref.project(project_to_plane)
        traj_est.project(project_to_plane)

    # Calculate APE.
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    if change_unit:
        ape_metric.change_unit(change_unit)

    ape_all_stats = ape_metric.get_all_statistics()
    title = str(ape_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"
    if (align or correct_scale) and n_to_align != -1:
        title += " (aligned poses: {})".format(n_to_align)

    if project_to_plane:
        title += f"\n(projected to {project_to_plane.value} plane)"

    ape_result = ape_metric.get_result(ref_name, est_name)
    ape_result.info["title"] = title

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, PoseTrajectory3D):
        seconds_from_start = np.array(
            [t - traj_est.timestamps[0] for t in traj_est.timestamps]
        )
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)
        ape_result.add_np_array("distances_from_start", traj_ref.distances)
        ape_result.add_np_array("distances", traj_est.distances)

    if alignment_transformation is not None:
        ape_result.add_np_array(
            "alignment_transformation_sim3", alignment_transformation
        )

    return ape_result, ape_all_stats, traj_est
