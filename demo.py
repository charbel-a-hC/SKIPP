import argparse
import copy
import json
import os
import random
import time
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from evo.tools import plot
from evo.tools.plot import PlotMode
from huggingface_hub import hf_hub_download
from PIL import Image

from bc.BCRunner import BCRunner
from configs.DotDict import DotDict
from dataset.ExpertFlatSimData import get_flatsim_data
from metrics.ape import calculate_ape_from_prediction
from metrics.fid import calculate_batch_fid
from utils.download_dataset import download_dataset
from utils.image_utils import (
    batch_transform,
    get_act_obs_reverse_transforms,
    get_start_end_reverse_transform,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run SKIPP model with different model shapes."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["skipp-l-shape", "skipp-u-shape"],
        default="skipp-l-shape",
        help="Model shape to use (skipp-l-shape or skipp-u-shape)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run prediction and FID calculation on",
    )
    return parser.parse_args()


def overlay_egm_path_custom(egm, path):
    out = copy.deepcopy(egm)
    out[..., 1] = path[..., 0] * 0.7 + out[..., 1]
    return out


def show_ape(result, traj_ref, traj_est, plot_mode, dict_res):
    fig = plt.figure(figsize=(7, 7))

    ax = plot.prepare_axis(fig, plot_mode)
    ax.grid(False)

    plot.traj(
        ax,
        plot_mode,
        traj_ref,
        style="--",
        alpha=0.5,
        plot_start_end_markers=True,
    )
    plot.traj_colormap(
        ax,
        traj_est,
        result.np_arrays["error_array"],
        plot_mode,
        min_map=result.stats["min"],
        max_map=result.stats["max"],
        plot_start_end_markers=True,
    )
    mean_ape, std_ape, min_ape, max_ape = (
        dict_res["mean"],
        dict_res["std"],
        dict_res["min"],
        dict_res["max"],
    )
    ape_stats_text = f"Mean APE: {mean_ape:.4f} m\nStd APE: {std_ape:.4f} m\nMin APE: {min_ape:.4f} m\nMax APE: {max_ape:.4f} m"
    ax.text(
        0.05,
        0.95,
        ape_stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    fig.savefig("assets/evo_output.png", dpi=150, bbox_inches="tight")


def save_overlay(
    egm, path, start, target, batch_idx, data_idx, title_prepend="gt"
):
    (
        act_reverse_transforms,
        obs_reverse_transforms,
    ) = get_act_obs_reverse_transforms(tensor=False)

    egm, path, start, target = egm.cpu(), path.cpu(), start.cpu(), target.cpu()

    egm_post = obs_reverse_transforms(egm[0, ...])
    path_post = act_reverse_transforms(path[0, ...])

    x_start, y_start = np.where(start[0, ...].cpu().numpy()[0, ...] == 1)
    x_start, y_start = x_start[0], y_start[0]

    x_end, y_end = np.where(target[0, ...].cpu().numpy()[0, ...] == 1)
    x_end, y_end = x_end[0], y_end[0]

    img = Image.fromarray(egm_post)
    img.save(f"assets/sample_egm.png")

    img = Image.fromarray(path_post)
    img.save(f"assets/{title_prepend}_path.png")

    egm_post[x_end - 5 : x_end + 5, y_end - 5 : y_end + 5, 2] = 255
    egm_post[x_start - 5 : x_start + 5, y_start - 5 : y_start + 5, 0] = 255

    egm_post_overlay = overlay_egm_path_custom(egm_post, path_post)

    img = Image.fromarray(egm_post_overlay)
    img.save(f"assets/{title_prepend}_overlay.png")


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
        binary_gt, binary_pred, start, target = (
            gt_path[i],
            pred[i],
            batch_start[i],
            batch_end[i],
        )
        output = calculate_ape_from_prediction(
            binary_gt, binary_pred, start, target
        )
        if output:
            (vis_res, res, traj_est), traj_ref = output
            for key, val in res.items():
                if not np.isnan(val):
                    results[key].append(val)
    if len(results["rmse"]) < 1:
        return None
    return {key: np.mean(val) for key, val in results.items()}, output


def load_model(config: dict, ret_config=True):
    bc_config = DotDict({"name": "", "project": "", "entity": ""})
    bc_runner = BCRunner(bc_config, 0)

    bc_runner.load_eval_model(config)
    if not ret_config:
        return bc_runner

    return bc_runner, config


def download_model(repo_id, filename="skipp-u-shape", cache_dir=None):
    """Download a model file from Hugging Face Hub"""
    print(f"Downloading model from {repo_id}...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        cache_dir=cache_dir,
    )

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{filename}.config.json",
        repo_type="model",
        cache_dir=cache_dir,
    )

    print(f"Model downloaded to {model_path}")
    return model_path, config_path


def main():
    args = parse_arguments()

    # Model repository selection based on model shape
    if args.model == "skipp-l-shape":
        repo_id = "charbel-a-h/SKIPP-L-Shape"
        task_shape = "L_Shape"
    else:
        repo_id = "charbel-a-h/SKIPP-U-Shape"
        task_shape = "U_Shape"

    # Download dataset
    test_dataset_path = download_dataset(
        repo_id="charbel-a-h/SKIPP-Expert-Data",
        cache_dir="expert_data/",
        data="test",
    )

    # Download model
    model_path, config_path = download_model(
        repo_id=repo_id,
        cache_dir=args.cache_dir or "models/",
        filename=args.model,
    )

    # Load model configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    config["path"] = model_path
    config["device"] = args.device

    set_seed(config["seed"])

    # Load dataset
    test_loader = get_flatsim_data(
        root_dir=f"{test_dataset_path}/test/{task_shape}",
        shuffle=True,
        batch_size=args.batch_size,
        nb_datapoints=-1,
        train_test_val_split=None,
        spline_path=True,
        data_seed=config["data_seed"],
        spline_points=1000,
        s=10,
    )

    # Load and evaluate model
    runner = load_model(config, False)
    runner.bc_model.eval()

    print(f"Successfully loaded {args.model} model")

    model_log = {}
    ape_batch = {
        "rmse": [],
        "mean": [],
        "median": [],
        "std": [],
        "min": [],
        "max": [],
        "sse": [],
    }

    inference_per_instance = []
    fid_per_batch = []

    # Randomize instance and batch as needed
    rand_batch = 0
    rand_idx = 30

    for i, sample in enumerate(test_loader):
        if i == rand_batch:
            egm, gt_a, start, target = [
                x[rand_idx, ...][None, ...].to(config["device"])
                for x in sample
            ]

            # REMOVE SAVING IF YOU WANT TO ITERATE OVER THE ENTIRE BATCH/DATASET
            save_overlay(
                egm=egm,
                path=gt_a,
                start=start,
                target=target,
                batch_idx=rand_batch,
                data_idx=rand_idx,
            )

            if config["obs"] == "semi":
                batch_s = torch.cat([egm, target], dim=1)
            elif config["obs"] == "full":
                batch_s = torch.cat([egm, start, target], dim=1)

            batch_s = batch_s.to(config["device"])
            t1 = time.time()
            bc_a = runner.bc_model(batch_s)
            inference_per_instance.append((time.time() - t1) * 1000)

            # REMOVE SAVING IF YOU WANT TO ITERATE OVER THE ENTIRE BATCH/DATASET
            save_overlay(
                egm=egm,
                path=bc_a.detach(),
                start=start,
                target=target,
                batch_idx=rand_batch,
                data_idx=rand_idx,
                title_prepend="pred",
            )

            prediction = bc_a.detach().cpu().numpy()
            prediction[prediction > 0.01] = 1
            prediction = np.asarray(prediction * 1, np.uint8)

            res, output = batch_ape_from_prediction(
                ground_truth=(gt_a, start, target),
                pred=prediction,
                align=False,
            )

            fid_per_batch.append(
                calculate_batch_fid(
                    bc_a.detach(), gt_a, device=config["device"]
                ).item()
            )
            if res:
                for key, value in ape_batch.items():
                    ape_batch[key].append(res[key])

    model_log.update(
        {f"{key}": np.mean(val) for key, val in ape_batch.items()}
    )
    model_log.update({"average_inference": np.mean(inference_per_instance)})
    model_log.update({"batch_fid": np.mean(fid_per_batch)})

    print(model_log)
    (vis_res, res, traj_est), traj_ref = output
    show_ape(vis_res, traj_ref, traj_est, PlotMode.xy, res)


if __name__ == "__main__":
    main()
