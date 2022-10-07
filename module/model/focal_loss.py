import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def binary_focal_loss(input: Tensor, target: Tensor, alpha=0.25, gamma=2, reduction='mean', pos_weight=None, is_logits=False) -> Tensor:
    
    if not pos_weight:
        pos_weight = torch.ones(input.shape[1], device=input.device, dtype=input.dtype)
    
    if not is_logits:
        p = input.sigmoid()
    else:
        p = input
        
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=pos_weight)
    
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
        
    
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', pos_weight=None, is_logits=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.is_logits = is_logits
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return binary_focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.pos_weight, self.is_logits)