import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = torch.tensor(gamma)
        self.reduction = reduction

    def forward(self, logits, targets, is_focal):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if not is_focal:
            if self.reduction == 'mean':
                return torch.mean(bce_loss)
            elif self.reduction == 'sum':
                return torch.sum(bce_loss)
            else:
                return bce_loss

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        self.alpha = self.alpha.to(logits.device)
        self.gamma = self.gamma.to(logits.device)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
