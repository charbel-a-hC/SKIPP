import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        # Apply sigmoid activation to the model's output
        input_sigmoid = torch.sigmoid(input)

        # Flatten the input and target tensors
        input_flat = input_sigmoid.view(-1)
        target_flat = target.view(-1)

        # Calculate the binary cross-entropy loss
        loss = F.binary_cross_entropy(
            input_flat, target_flat, reduction="none"
        )

        # Weighting the positive class
        if self.alpha is not None:
            alpha_tensor = torch.full_like(target_flat, self.alpha)
            alpha_tensor[target_flat == 0] = 1 - self.alpha
            loss = alpha_tensor * loss

        # Calculate the mean loss
        loss = torch.mean(loss)

        return loss


class BinaryDiceLoss(torch.nn.Module):
    """
    Binary Dice Loss for image segmentation tasks with a single class.
    """

    def __init__(self, epsilon=1e-6):
        super(BinaryDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        """
        Compute the Binary Dice Loss between prediction and target.

        Args:
            prediction (torch.Tensor): Predicted segmentation mask (batch_size, 1, height, width)
            target (torch.Tensor): Ground truth segmentation mask (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Binary Dice Loss
        """
        intersection = torch.sum(prediction * target)
        cardinality_prediction = torch.sum(prediction)
        cardinality_target = torch.sum(target)

        dice = (2.0 * intersection + self.epsilon) / (
            cardinality_prediction + cardinality_target + self.epsilon
        )

        return 1 - dice


LOSSES = {
    "mse": {"func": nn.MSELoss, "parameters": {"reduction": "mean"}},
    "l1": {"func": nn.L1Loss, "parameters": {"reduction": "mean"}},
    "bce": {"func": nn.BCELoss, "parameters": {"reduction": "mean"}},
    "weighted-bce": {
        "func": WeightedBinaryCrossEntropyLoss,
        "parameters": {"alpha": 0.5},
    },
    "dice": {"func": BinaryDiceLoss, "parameters": {"epsilon": 1e-6}},
}
