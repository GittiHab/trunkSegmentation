import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class BinaryWeightedDiceBCELoss(nn.Module):
    def __init__(self, weight: float = 0.25):
        # default weight based on "Learning to Correct Sloppy Annotations in Electron Microscopy Volumes" by Chen et al.
        # https://openaccess.thecvf.com/content/CVPR2023W/CVMI/papers/Chen_Learning_To_Correct_Sloppy_Annotations_in_Electron_Microscopy_Volumes_CVPRW_2023_paper.pdf
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCELoss()
        self.dice = torchmetrics.functional.dice

    def forward(self, input, targets):
        return self.weight * (1 - self.dice(input, targets)) + (1 - self.weight) * self.bce_loss(input, targets.float())

class FocalLossCustom(nn.Module):
    """
    Focal loss implementation for binary classification tasks.

    Focal loss is designed to address the issue of class imbalance by focusing
    on challenging examples during training. It downweights easy examples and
    emphasizes misclassified examples.

    The focal loss formula is given by:
    focal_loss = -α * (1 - pt)^γ * log(pt)

    Args:
        gamma (float): The focusing parameter that controls the degree of
            focusing or emphasis on hard examples. Higher values of gamma
            increase the focus on hard examples. Default is 2.0.
        alpha (float): The balance parameter that adjusts the weight given to
            different classes. It addresses class imbalance by assigning
            higher weights to minority class examples. Default is 0.5.
        logits (bool): Flag indicating whether the inputs are logits (raw
            unnormalized scores) or probabilities. If True, inputs are assumed
            to be logits; if False, inputs are assumed to be probabilities.
            Default is True.
        reduction (str): Specifies the reduction method for the computed loss.
            Options are 'sum', 'mean', or 'none'. Default is 'mean'.


    Inputs:
        inputs (torch.Tensor): The model predictions or logits.
        targets (torch.Tensor): The ground truth labels.

    Returns:
        torch.Tensor: The computed focal loss.

    Shape:
        - inputs: (batch_size, ...)
        - targets: (batch_size, ...)
        - Output: Scalar value.

    Thank you to ChatGPT (GPT 3.5) for aiding the implementation.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, logits: bool = True, reduction: str = 'mean'):
        super(FocalLossCustom, self).__init__()
        assert reduction in ['mean', 'none',
                             'sum'], f"Invalid reduction method '{self.reduction}'. Choose from 'sum', 'mean', or 'none'."

        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * torch.pow(1 - pt, self.gamma) * bce_loss
        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'none':
            return focal_loss