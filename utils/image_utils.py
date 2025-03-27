import numpy as np
import torch
import torchvision.transforms as T


def post_process_act(self, action):
    act_reverse_transforms = T.Compose(
        [
            T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            T.Lambda(lambda t: t * 255.0),
            T.Lambda(lambda x: x.repeat(1, 1, 3) if x.size(2) == 1 else x),
            T.Lambda(lambda t: t.numpy().astype(np.uint8)),
            T.ToTensor(),
        ]
    )

    return act_reverse_transforms(action)


def threshold_obs(arr, lower_val=60, upper_val=140):
    arr[arr <= lower_val] = 0
    arr[np.where((arr > lower_val) & (arr < upper_val))] = 75
    arr[arr >= upper_val] = 255
    return arr


def reverse_normalize(
    output_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    # Convert tensor to numpy array
    output_np = output_tensor.detach().cpu()

    # Reshape mean and std to match the shape of the tensor
    mean = torch.tensor(mean).reshape(-1, 1, 1)
    std = torch.tensor(std).reshape(-1, 1, 1)

    # Reverse normalization

    output_np = (output_np * std) + mean

    return output_np


def batch_transform(batch_tensor, transform):
    out_arr = []
    for i, tensor in enumerate(batch_tensor):
        transformed_tensor = transform(tensor)
        if isinstance(transformed_tensor, torch.Tensor):
            transformed_tensor = transformed_tensor.unsqueeze(0)
        elif isinstance(transformed_tensor, np.ndarray):
            transformed_tensor = np.expand_dims(transformed_tensor, 0)
        out_arr.append(transformed_tensor)

    if isinstance(transformed_tensor, torch.Tensor):
        return torch.vstack(out_arr)
    return np.vstack(out_arr)


def overlay_egm_path(egm, gt_path, pred_path):
    # egm[:, 0, ...] = gt_path[:, 0, ...] + egm[:, 0, ...]
    egm[:, 1, ...] = pred_path[:, 0, ...] + egm[:, 1, ...]

    return egm


def get_act_obs_reverse_transforms(tensor=True):
    act_reverse_transforms = T.Compose(
        [
            T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            T.Lambda(lambda t: t * 255.0),
            T.Lambda(lambda x: x.repeat(1, 1, 3) if x.size(2) == 1 else x),
            T.Lambda(lambda t: t.numpy().astype(np.uint8)),
        ]
    )

    obs_reverse_transforms = T.Compose(
        [
            T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            T.Lambda(
                lambda t: reverse_normalize(t, mean=[0.485], std=[0.229])
            ),
            T.Lambda(lambda x: x.repeat(1, 1, 3) if x.size(2) == 1 else x),
            T.Lambda(lambda t: t.numpy().astype(np.uint8)),
        ]
    )

    if tensor:
        act_reverse_transforms.transforms.append(T.ToTensor())
        obs_reverse_transforms.transforms.append(T.ToTensor())

    return act_reverse_transforms, obs_reverse_transforms


def get_start_end_reverse_transform(tensor=True):
    start_end_reverse_transforms = T.Compose(
        [
            T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            T.Lambda(lambda t: t.numpy().astype(np.uint8)),
            T.Lambda(lambda t: np.squeeze(t, 2)),
        ]
    )

    if tensor:
        start_end_reverse_transforms.transforms.append(T.ToTensor())

    return start_end_reverse_transforms
