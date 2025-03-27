import json
import os
import random
import sys
import traceback

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torchvision import utils

import wandb
from bc.losses import LOSSES
from configs import BCSweepConfig
from configs.DotDict import DotDict
from dataset.ExpertFlatSimData import get_flatsim_data
from metrics.ape import batch_ape_from_prediction
from metrics.fid import calculate_batch_fid
from utils.image_utils import (
    batch_transform,
    get_act_obs_reverse_transforms,
    overlay_egm_path,
)
from utils.logger import get_logger

from .UNetSkip import UnetBC


class BCRunner:
    def __init__(self, bc_config: BCSweepConfig, sweep_id: int):
        self.project, self.entity = bc_config.project, bc_config.entity
        self.sweep_id = sweep_id

        self.logger = get_logger(description="BCRunner")
        self.loss_logs = {}

        self.bc_config = bc_config

    def collate_batch_images(self, obs_act_arr):
        grid = utils.make_grid(obs_act_arr, nrow=3)
        npgrid = grid.numpy()

        collated = np.transpose(npgrid, (1, 2, 0))

        return collated

    def set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        if "cuda" in self.bc_config.device:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def setup_models(self, wandb_config):
        # Setup BC model
        self.logger.debug("Setting up BC model ...")
        self.bc_model = UnetBC.from_config(
            wandb_config,
            out_channels=self.bc_config.bc_backbone_params.out_channels,
        ).to(self.bc_config.device)
        self.logger.debug(f"Model loaded!")

    def create_dirs(self, run_name):
        self.run_path = f"runs/{self.sweep_id}/{run_name}"
        os.makedirs(self.run_path)

    def train_step(self, data_loader, obs):
        loss_log = {}
        self.bc_model.train()

        bc_batch = []
        ape_batch = {
            "rmse": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            "sse": [],
        }

        rand_act, rand_pred, rand_start, rand_target = None, None, None, None
        idx = np.random.randint(0, len(data_loader))

        for i, sample in enumerate(data_loader):
            self.optimizer.zero_grad()

            batch_s, gt_a, start, target = [
                x.to(self.bc_config.device) for x in sample
            ]

            if obs == "semi":
                bc_batch_s = torch.cat([batch_s, target], dim=1)

            elif obs == "full":
                bc_batch_s = torch.cat([batch_s, start, target], dim=1)

            elif obs == "slim":
                bc_batch_s = batch_s

            bc_a = self.bc_model(bc_batch_s)
            l_bc = self.loss_func(bc_a, gt_a)

            if i == idx:
                rand_act = gt_a
                rand_pred = bc_a
                rand_start = start
                rand_target = target

            l_bc.backward()

            self.optimizer.step()
            bc_batch.append(l_bc.item())

        loss_log["bc"] = np.mean(bc_batch)
        loss_log["fid"] = calculate_batch_fid(
            rand_pred.detach(), rand_act, device=self.bc_config.device
        )

        prediction = rand_pred.detach().cpu().numpy()
        prediction[prediction > 0.01] = 1
        prediction = np.asarray(prediction * 1, np.uint8)

        res = batch_ape_from_prediction(
            ground_truth=(rand_act, rand_start, rand_target), pred=prediction
        )
        if res:
            ape_batch = res
        loss_log.update(
            {f"ape_{key}": np.mean(val) for key, val in ape_batch.items()}
        )

        return loss_log

    @torch.no_grad()
    def val_step(self, data_loader, obs):
        loss_log = {}

        bc_batch = []
        ape_batch = {
            "rmse": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            "sse": [],
        }

        rand_act, rand_pred, rand_start, rand_target = None, None, None, None
        idx = np.random.randint(0, len(data_loader))

        for i, sample in enumerate(data_loader):
            self.optimizer.zero_grad()
            batch_s, gt_a, start, target = [
                x.to(self.bc_config.device) for x in sample
            ]

            if obs == "semi":
                bc_batch_s = torch.cat([batch_s, target], dim=1)

            elif obs == "full":
                bc_batch_s = torch.cat([batch_s, start, target], dim=1)

            elif obs == "slim":
                bc_batch_s = batch_s

            bc_a = self.bc_model(bc_batch_s)
            l_bc = self.loss_func(bc_a, gt_a)

            if i == idx:
                rand_act = gt_a
                rand_pred = bc_a
                rand_start = start
                rand_target = target

            bc_batch.append(l_bc.item())

        loss_log["bc"] = np.mean(bc_batch)

        prediction = rand_pred.detach().cpu().numpy()
        prediction[prediction > 0.01] = 1
        prediction = np.asarray(prediction * 1, np.uint8)

        res = batch_ape_from_prediction(
            ground_truth=(rand_act, rand_start, rand_target), pred=prediction
        )
        if res:
            ape_batch = res
        loss_log.update(
            {f"ape_{key}": np.mean(val) for key, val in ape_batch.items()}
        )

        loss_log["fid"] = calculate_batch_fid(
            rand_pred.detach(), rand_act, device=self.bc_config.device
        )
        return loss_log

    @torch.no_grad()
    def test_step(self, data_loader, obs):
        loss_log = {}
        bc_batch = []

        ape_batch = {
            "rmse": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            "sse": [],
        }

        rand_act, rand_pred, rand_start, rand_target = None, None, None, None
        idx = np.random.randint(0, len(data_loader))

        for i, sample in enumerate(data_loader):
            self.optimizer.zero_grad()
            batch_s, gt_a, start, target = [
                x.to(self.bc_config.device) for x in sample
            ]

            if obs == "semi":
                bc_batch_s = torch.cat([batch_s, target], dim=1)

            elif obs == "full":
                bc_batch_s = torch.cat([batch_s, start, target], dim=1)

            elif obs == "slim":
                bc_batch_s = batch_s

            bc_a = self.bc_model(bc_batch_s)
            l_bc = self.loss_func(bc_a, gt_a)

            if i == idx:
                rand_act = gt_a
                rand_pred = bc_a
                rand_start = start
                rand_target = target

            bc_batch.append(l_bc.item())

        prediction = rand_pred.detach().cpu().numpy()
        prediction[prediction > 0.01] = 1
        prediction = np.asarray(prediction * 1, np.uint8)

        res = batch_ape_from_prediction(
            ground_truth=(rand_act, rand_start, rand_target), pred=prediction
        )
        if res:
            ape_batch = res
        loss_log.update(
            {f"ape_{key}": np.mean(val) for key, val in ape_batch.items()}
        )

        loss_log["fid"] = calculate_batch_fid(
            rand_pred.detach(), rand_act, device=self.bc_config.device
        )
        return loss_log

    def train(self):
        try:
            run = wandb.init(project=self.project, entity=self.entity)

            wandb.run.log_code("../")
            sweep_config = wandb.config

            self.set_seed(sweep_config.seed)
            run_name = run.name

            self.create_dirs(run_name=wandb.run.name)

            # Setup dataset
            data_loader = get_flatsim_data(
                root_dir=self.bc_config.root_dir,
                resize=tuple(self.bc_config.resize),
                nb_datapoints=self.bc_config.nb_datapoints,
                batch_size=sweep_config.batch_size,
                train_test_val_split=sweep_config.train_test_val_split,
                spline_path=sweep_config.spline_path,
                spline_points=self.bc_config.spline_points,
                s=self.bc_config.s,
                shuffle=self.bc_config.shuffle,
                data_seed=sweep_config.data_seed,
            )

            # Setup loss function
            func = LOSSES[sweep_config.loss]["func"]
            self.loss_func = func(**LOSSES[sweep_config.loss]["parameters"])

            # Setup Models
            self.setup_models(sweep_config)

            self.optimizer = optim.Adam(
                self.bc_model.parameters(), lr=sweep_config.lr
            )

            for t in range(sweep_config.epochs):
                train_loss_log = self.train_step(
                    data_loader=data_loader[0], obs=sweep_config.obs
                )
                val_loss_log = self.val_step(
                    data_loader[1], obs=sweep_config.obs
                )

                log_msg = f"Epoch: {t+1}/{sweep_config.epochs}"
                for key, val in train_loss_log.items():
                    log_msg += f" train_{key}_loss: {val:.4f}"
                    if self.bc_config.wandb_log_interval:
                        wandb.log({f"train/{key}": val})

                for key, val in val_loss_log.items():
                    log_msg += f" val_{key}_loss: {val:.4f}"
                    if self.bc_config.wandb_log_interval:
                        wandb.log({f"val/{key}": val})

                self.logger.info(log_msg)

                self.loss_logs.update(
                    {
                        t: {
                            "train_loss": train_loss_log,
                            "val_loss": val_loss_log,
                        }
                    }
                )

                if t % self.bc_config.save_interval == 0:
                    self.save_model(path=f"{self.run_path}/{run_name}_{t}")

            if len(data_loader) == 3:
                test_loss_log = self.test_step(
                    data_loader[2], obs=sweep_config.obs
                )

                for key, val in test_loss_log.items():
                    # log_msg += f" test_{key}_loss: {val:.4f}"
                    if self.bc_config.wandb_log_interval:
                        wandb.log({f"test/{key}": val})

                self.loss_logs.update({t: {"test_loss": test_loss_log}})
            # self.logger.info(log_msg)

            try:
                for batch_s, gt_a, start, target in data_loader[1]:
                    batch_indices = torch.randint(
                        size=(5,), high=batch_s.shape[0], low=0
                    )
                    self.logger.info(
                        f"Batch indices for posting: {batch_indices}"
                    )

                    collated_predictions = self.get_image_predictions(
                        sample_egm=batch_s[batch_indices, ...],
                        gt=gt_a[batch_indices, ...],
                        end=target[batch_indices, ...],
                        start=start[batch_indices, ...],
                        obs=sweep_config.obs,
                        device=self.bc_config.device,
                    )

                    wandb.log(
                        {
                            "Input EGM Overlayed with Predicted Path + Ground Truth Path (2nd Row)": wandb.Image(
                                collated_predictions
                            )
                        }
                    )

                    img = Image.fromarray(collated_predictions)
                    img.save(f"{self.run_path}/{run_name}_losses.jpg")
            finally:
                self.save_model(path=f"{self.run_path}/{run_name}_final")
                self.save_losses(
                    path=f"{self.run_path}/{run_name}_losses.json"
                )

        except Exception:
            print(traceback.print_exc(), file=sys.stderr)

    def save_model(self, path="bc-unet-large"):
        torch.save(self.bc_model.state_dict(), path)

    def save_losses(self, path="bc-unet-large"):
        with open(path, "w") as f:
            json.dump(self.loss_logs, f, indent=4)

    def _load_model(self, path="bc-unet-large", device="cuda"):
        self.bc_model.load_state_dict(torch.load(path, device))

    def load_eval_model(self, config: dict):
        dot_config = DotDict(config)
        out_channels = dot_config.bc_backbone_params.out_channels
        self.bc_model = UnetBC.from_config(
            dot_config, out_channels=out_channels
        ).to(dot_config.device)

        self._load_model(path=dot_config.path, device=dot_config.device)

    def get_image_predictions(
        self,
        sample_egm,
        gt,
        end,
        start,
        obs,
        device,
        return_loss=False,
        loss_fn="bce",
    ):
        batch_s = sample_egm.to(device)

        if obs == "semi":
            batch_s = torch.cat([batch_s, end.to(device)], dim=1)

        elif obs == "full":
            batch_s = torch.cat(
                [batch_s, start.to(device), end.to(device)], dim=1
            )

        pred = self.bc_model(batch_s).detach()

        (
            act_reverse_transforms,
            obs_reverse_transforms,
        ) = get_act_obs_reverse_transforms()
        pred_post = batch_transform(
            pred.cpu().squeeze(0), act_reverse_transforms
        )
        gt_post = batch_transform(gt, act_reverse_transforms)
        egm_post = batch_transform(sample_egm, obs_reverse_transforms)
        egm_post = overlay_egm_path(egm_post, gt_post, pred_post)

        target_idx = np.where(end[:, ...] == 1)
        start_idx = np.where(start[:, ...] == 1)
        # add start and end points
        for i in range(batch_s.shape[0]):
            x, y = target_idx[2][i], target_idx[3][i]
            x_start, y_start = start_idx[2][i], start_idx[3][i]
            egm_post[i, 2, x - 5 : x + 5, y - 5 : y + 5] = 1.0
            egm_post[
                i, 0, x_start - 5 : x_start + 5, y_start - 5 : y_start + 5
            ] = 1.0

        # color prediction in green
        pred_post[:, 0, ...] = torch.zeros_like(pred_post[:, 0, ...])
        pred_post[:, 2, ...] = torch.zeros_like(pred_post[:, 0, ...])

        # color GT in some other color
        gt_post[:, 1, ...] = torch.zeros_like(gt_post[:, 0, ...])

        collated_data = torch.vstack((egm_post, gt_post, pred_post))
        grid = utils.make_grid(collated_data, nrow=5)
        npgrid = grid.numpy()

        collated_images = np.transpose(npgrid, (1, 2, 0))
        collated_images = (collated_images * 255).astype(np.uint8)

        if return_loss:
            loss_fn = LOSSES[loss_fn]["func"]()
            loss_val = loss_fn(pred, gt.to(device)).item()
            self.logger.debug(f"Batch loss: {loss_val}")
            return collated_images, loss_val

        return collated_images
