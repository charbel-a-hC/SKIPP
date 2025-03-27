import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.interpolate import splev, splprep
from skimage.morphology import thin
from torch.utils.data import Dataset

from metrics.ape import convert_image_to_path
from utils.path_utils import extract_path_data, generate_goal_poses


def threshold_obs(arr, lower_val=60, upper_val=140):
    arr[arr <= lower_val] = 0
    arr[np.where((arr > lower_val) & (arr < upper_val))] = 75
    arr[arr >= upper_val] = 255
    return arr


def reset_start_target_vals(arr):
    arr[arr > 0] = 1.0
    return arr


def get_flatsim_data(
    root_dir="",
    resize=(256, 256),
    shuffle=False,
    batch_size=32,
    nb_datapoints=-1,
    train_test_val_split=[0.9, 0.10],
    spline_path=False,
    spline_points=100,
    s=100,
    data_seed=42,
):
    obs_transforms = T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Lambda(lambda t: t * 255),
            # T.Lambda(lambda t: np.asarray(t.numpy(), dtype=np.uint8)),
            T.Lambda(lambda t: t.to(torch.uint8)),
            T.Lambda(lambda t: threshold_obs(t)),
            T.Lambda(lambda t: t.float()),
            T.Normalize(mean=[0.485], std=[0.229])
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # T.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1],
            # T.Lambda(lambda t: t / 2 + 0.5),  # Scale between [0, 1]
        ]
    )

    acts_transforms = T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            # T.Normalize((0.5, ), (0.5, ))
        ]
    )

    egm_target_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(resize, interpolation=T.InterpolationMode.BOX),
            T.ToTensor(),
            T.Lambda(lambda t: reset_start_target_vals(t)),
        ]
    )

    flatsim_expert_dataset = ExpertFlatSimData(
        root_dir,
        obs_transforms=obs_transforms,
        acts_transforms=acts_transforms,
        egm_target_transform=egm_target_transform,
        nb_datapoints=nb_datapoints,
        spline_path=spline_path,
        spline_points=spline_points,
        s=s,
    )

    if not train_test_val_split:
        train_loader = torch.utils.data.DataLoader(
            dataset=flatsim_expert_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return train_loader

    elif len(train_test_val_split) == 2:
        assert sum(train_test_val_split) <= 1, "Split should sum to 1.0"
        train_size, val_size = list(
            map(
                lambda t: int(len(flatsim_expert_dataset) * t),
                train_test_val_split,
            )
        )

        if len(flatsim_expert_dataset) != sum([train_size, val_size]):
            train_size += 1

        train_set, val_set = torch.utils.data.random_split(
            flatsim_expert_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(data_seed),
        )

        train_loader_flatsim = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=shuffle
        )

        val_loader_flatsim = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=batch_size, shuffle=shuffle
        )

        return train_loader_flatsim, val_loader_flatsim

    elif len(train_test_val_split) == 3:
        assert sum(train_test_val_split) <= 1, "Split should sum to 1.0"

        train_size, val_size, test_size = list(
            map(
                lambda t: int(len(flatsim_expert_dataset) * t),
                train_test_val_split,
            )
        )

        if len(flatsim_expert_dataset) != sum(
            [train_size, val_size, test_size]
        ):
            train_size += 1

        train_set, val_set, test_set = torch.utils.data.random_split(
            flatsim_expert_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(data_seed),
        )

        train_loader_flatsim = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=shuffle
        )
        val_loader_flatsim = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=batch_size, shuffle=shuffle
        )
        test_loader_flatsim = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=shuffle
        )
        return train_loader_flatsim, val_loader_flatsim, test_loader_flatsim


class ExpertFlatSimData(Dataset):
    def __init__(
        self,
        root_dir,
        obs_transforms=None,
        acts_transforms=None,
        egm_target_transform=None,
        nb_datapoints=-1,
        spline_path=False,
        spline_points=100,
        s=100,
    ):
        egm_naming = "_egm.jpg"
        path_naming = "_path.png"
        egm_goal_pose = "egm_goal_poses.npy"

        self.images_path = []
        self.obs_paths = []
        self.act_paths = []
        self.egm_goal_pose_path = []
        # create a loader that goes over all the images recursivly in all directories
        for root, dirs, files in os.walk(root_dir, topdown=True):
            if files and (root is not root_dir):
                self.obs_paths += [
                    f"{root}/{i+1}{egm_naming}"
                    for i in range(self.get_seq_len(root) - 1)
                ]
                self.act_paths += [
                    f"{root}/{i+1}{path_naming}"
                    for i in range(self.get_seq_len(root) - 1)
                ]
                self.egm_goal_pose_path.append(f"{root}/{egm_goal_pose}")

        if nb_datapoints > 0:
            self.obs_paths = self.obs_paths[:nb_datapoints]
            self.act_paths = self.act_paths[:nb_datapoints]
            self.egm_goal_pose_path = self.egm_goal_pose_path[:nb_datapoints]

        self.transforms = [
            obs_transforms,
            acts_transforms,
            egm_target_transform,
        ]
        self.egm_poses = np.concatenate(
            [np.load(a) for a in self.egm_goal_pose_path]
        )[
            :, :2
        ]  # EGM size, should be changed  if size changes
        self.spline_points = spline_points
        self.s = s
        self.spline_path = spline_path

    def get_seq_len(self, path: str):
        return len([x for x in os.listdir(path) if "egm" in x])

    def open_action_image(self, image_path):
        l_image = Image.open(image_path).convert("L")
        return l_image.point(lambda x: 0 if x < 128 else 255, "1")

    def remove_redundant_data(self):
        similar_images = []

        prev_image = np.array(
            self.open_action_image(self.act_paths[-1]), dtype=np.uint8
        )

        for i, image_path in enumerate(reversed(self.act_paths[:-1])):
            curr_image = np.array(
                self.open_action_image(image_path), dtype=np.uint8
            )

            matches, _ = np.where(curr_image != prev_image)

            if matches.shape[0] < 1:
                idx = len(self.act_paths) - i
                similar_images.append(
                    (
                        f"{self.images_path}/{idx}_path.png",
                        f"{self.images_path}/{idx}.png",
                    )
                )

        for act_path, obs_path in similar_images:
            self.obs_paths.remove(obs_path)
            self.act_paths.remove(act_path)
            os.remove(act_path)
            os.remove(obs_path)

    def spline_act(self, act, start, end, spline_points=100, s=100):
        convert_to_pixels = lambda t: int(t / 0.01)
        smooth_act_image, smooth_start, smooth_end = list(
            map(np.zeros_like, [act, start, end])
        )

        # # convert act image to path
        # thinned_act = thin(np.array(act))

        # path: List[dict]  = generate_goal_poses(extract_path_data(start, end, thinned_act))
        # path_arr = np.array([np.array([data["x"], data["y"]]) for data in path])

        # act, start, end = tuple(map(np.array, [act, start, end]))
        act = act.convert("RGB")
        act = np.array(act)[..., 0]

        path_arr = convert_image_to_path(
            np.asarray(act / 255.0, dtype=np.uint8),
            np.asarray(start / 255.0, dtype=np.uint8),
            np.asarray(end / 255.0, dtype=np.uint8),
            pred=True,
        )
        if len(path_arr) <= 3:
            act = Image.fromarray(act)
            act = act.point(lambda x: 0 if x < 128 else 255, "1")
            return act, start, end

        # smooth_path = resample_path_u(path_arr, spline_points, smoothing=s, k=3)
        smooth_path = resample_path(
            path_arr, spline_points, min(s, len(path_arr))
        )

        for x, y in smooth_path:
            smooth_act_image[convert_to_pixels(y), convert_to_pixels(x)] = 1

        start_x, start_y = convert_to_pixels(
            smooth_path[0, 1]
        ), convert_to_pixels(smooth_path[0, 0])
        if not (start_y <= 126 + 20 and start_y >= 126 - 20):
            smooth_start[
                convert_to_pixels(smooth_path[-1, 1]),
                convert_to_pixels(smooth_path[-1, 0]),
            ] = 1
            smooth_end[
                convert_to_pixels(smooth_path[0, 1]),
                convert_to_pixels(smooth_path[0, 0]),
            ] = 1
        else:
            smooth_start[
                convert_to_pixels(smooth_path[0, 1]),
                convert_to_pixels(smooth_path[0, 0]),
            ] = 1
            smooth_end[
                convert_to_pixels(smooth_path[-1, 1]),
                convert_to_pixels(smooth_path[-1, 0]),
            ] = 1

        pil_smooth: Image = Image.fromarray(smooth_act_image)

        return pil_smooth, smooth_start, smooth_end

    def __getitem__(self, index):
        obs = Image.open(self.obs_paths[index]).convert("L")
        act = Image.open(self.act_paths[index]).convert("L")
        act = act.point(lambda x: 0 if x < 128 else 255, "1")

        indices = self.egm_poses[index]
        egm_target = np.zeros(shape=obs.size, dtype=np.float32)
        egm_target[indices[0], indices[1]] = 1.0

        egm_start = np.zeros(shape=obs.size, dtype=np.float32)
        egm_start[62:65, 126:129] = 1.0

        # TODO Fix this mess
        # Spline path
        if self.spline_path:
            act, egm_start, egm_target = self.spline_act(
                act, egm_start, egm_target, self.spline_points, self.s
            )

        if all(self.transforms):
            obs = self.transforms[0](obs)
            act = self.transforms[1](act)
            egm_target = self.transforms[2](egm_target)
            egm_start = self.transforms[2](egm_start)

        else:
            obs = np.array(obs)
            act = np.array(act, dtype=np.uint8)
        return (obs, act, egm_start, egm_target)

    def __len__(self):
        return len(self.obs_paths)


def resample_path(path, num_points, s=100):
    # Get the x and y coordinates from the path
    x = path[:, 0]
    y = path[:, 1]

    # Parameterize the path with respect to a normalized variable t (0 to 1)
    t = np.linspace(0, 1, len(path))

    # Perform spline interpolation for the path (Cubic spline)
    # s=0 ensures no smoothing of the points (pure interpolation)
    tck, u = splprep([x, y], s=s)

    # Create a new set of num_points points equally spaced along the parameter t
    t_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(t_new, tck)

    # Create the new interpolated path
    new_path = np.vstack((x_new, y_new)).T

    return new_path


def calculate_cumulative_distances(path):
    diff = np.diff(path, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.concatenate(([0], np.cumsum(segment_lengths)))
    return cumulative_distances


def resample_path_u(path, num_points, smoothing=0, k=3):
    # Get the x and y coordinates from the path
    x = path[:, 0]
    y = path[:, 1]

    # Calculate the cumulative distance along the path
    distances = calculate_cumulative_distances(path)
    total_distance = distances[-1]

    # Normalize the distances to be between 0 and 1
    u = distances / total_distance

    # Perform spline interpolation for the path
    tck, _ = splprep([x, y], u=u, s=smoothing, k=k)

    # Create a new set of num_points points equally spaced along the path
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    # Create the new interpolated path
    new_path = np.vstack((x_new, y_new)).T

    return new_path


if __name__ == "__main__":
    sample_dataset = ExpertFlatSimData(root_dir="expert_data/dataset/")
