import numpy as np
import torch
from numpy import cov, iscomplexobj, trace
from torchvision import transforms

from utils.image_utils import batch_transform, get_act_obs_reverse_transforms


@torch.no_grad()
def calculate_batch_fid(
    images1,
    images2,
    device,
    diff_docking_pre_process=True,
    inception_pre_process=True,
):
    """All images must be pre-processed to the

    Args:
        images1 (_type_): _description_
        images2 (_type_): _description_
    """

    def matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
        vals, vecs = torch.eig(matrix, eigenvectors=True)
        vals = torch.view_as_complex(vals.contiguous())
        vals_pow = vals.pow(p)
        vals_pow = torch.view_as_real(vals_pow)[:, 0]
        matrix_pow = torch.matmul(
            vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs))
        )
        return matrix_pow

    def load_inception_model():
        import torchvision.models as models

        # Load the InceptionV3 model
        model = models.inception_v3(pretrained=True)

        # Remove the fully connected head
        model.fc = torch.nn.Identity()
        model.dropout = torch.nn.Identity()

        # Set the model to evaluation mode
        model.eval()
        return model

    inception_model = load_inception_model().to(device)

    if diff_docking_pre_process:
        act_reverse_transforms, _ = get_act_obs_reverse_transforms()
        images1, images2 = batch_transform(
            batch_tensor=images1.cpu(), transform=act_reverse_transforms
        ), batch_transform(
            batch_tensor=images2.cpu(), transform=act_reverse_transforms
        )

    if inception_pre_process:
        inception_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        images1 = batch_transform(
            batch_tensor=images1, transform=inception_transforms
        ).to(device)
        images2 = batch_transform(
            batch_tensor=images2, transform=inception_transforms
        ).to(device)

    # calculate activations
    act1 = inception_model(images1).detach()
    act2 = inception_model(images2).detach()

    # Calculate mean and covariance on GPU
    mu1 = torch.mean(act1, dim=0)
    mu2 = torch.mean(act2, dim=0)

    # Covariance calculation (rowvar=False in numpy is the default behavior in PyTorch)
    if images1.shape[0] == 1:
        sigma1 = torch.var(act1)
        sigma2 = torch.var(act2)
    else:
        sigma1 = torch.cov(act1.T)  # Transpose to match shape
        sigma2 = torch.cov(act2.T)

    # Calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between covariances (using sqrtm for matrix square root)
    if images1.shape[0] == 1:
        covmean = torch.sqrt(sigma1 * sigma2)
    else:
        import torch.linalg as linalg

        U, s, Vh = linalg.svd(sigma1 @ sigma2)
        covmean = U @ torch.diag(torch.sqrt(s)) @ Vh
        # covmean = matrix_pow(sigma1 @ sigma2, 0.5).to(device)
    # Calculate the FID score
    if images1.shape[0] == 1:
        fid = ssdiff + torch.sum(sigma1) + torch.sum(sigma2) - 2.0 * covmean
        return fid
    else:
        fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid.item()
