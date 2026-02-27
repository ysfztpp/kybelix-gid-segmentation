import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_outputs import split_model_outputs

class FocalTverskyLoss(nn.Module):
    """
    Dengesiz sınıflar için (örneğin arazideki az bulunan yeşil alanlar) 
    optimize edilmiş kayıp fonksiyonu.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha   # False Negative ağırlığı
        self.beta = beta     # False Positive ağırlığı
        self.gamma = gamma   # Odaklanma parametresi
        self.smooth = smooth

    def forward(self, inputs, targets):
        # ÖNEMLİ: Model logits döndüğü için burada sigmoid uyguluyoruz
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()    
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)  
        return (1 - tversky) ** self.gamma


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        if num_classes == 1:
            probs = torch.sigmoid(logits)
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            if probs.shape != targets.shape:
                targets = targets.view_as(probs)

            intersection = (probs * targets).sum(dim=(0, 2, 3))
            cardinality = (probs + targets).sum(dim=(0, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            return 1.0 - dice.mean()

        if targets.dim() == 4:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
            else:
                targets = targets.argmax(dim=1)
        targets = targets.long()

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def build_edge_targets(masks, method="sobel", sobel_threshold=0.1):
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    if masks.dim() == 4 and masks.shape[1] > 1:
        masks = masks.argmax(dim=1, keepdim=True)
    masks = masks.float()

    if method == "morph":
        dilated = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)
        return ((dilated - eroded) > 0).float()

    if method == "sobel":
        sobel_x = masks.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
        sobel_y = masks.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(masks, sobel_x, padding=1)
        grad_y = F.conv2d(masks, sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return (magnitude > sobel_threshold).float()

    raise ValueError(f"Unknown edge target method: {method}. Use 'sobel' or 'morph'.")


class DiceCrossEntropyBoundaryLoss(nn.Module):
    """
    Total Loss = Dice(seg) + CrossEntropy(seg) + lambda * BCE(edge)
    """

    def __init__(self, lambda_edge=0.4, edge_method="sobel", sobel_threshold=0.1):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.edge_method = edge_method
        self.sobel_threshold = sobel_threshold

        self.dice_loss = DiceLoss()
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.ce_seg = nn.CrossEntropyLoss()
        self.bce_edge = nn.BCEWithLogitsLoss()

    def _cross_entropy_seg(self, seg_logits, targets):
        if seg_logits.shape[1] == 1:
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            if seg_logits.shape != targets.shape:
                targets = targets.view_as(seg_logits)
            return self.bce_seg(seg_logits, targets)

        if targets.dim() == 4:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
            else:
                targets = targets.argmax(dim=1)
        targets = targets.long()
        return self.ce_seg(seg_logits, targets)

    def forward(self, outputs, targets):
        seg_logits, edge_logits = split_model_outputs(outputs)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        target_size = targets.shape[-2:]
        if seg_logits.shape[-2:] != target_size:
            seg_logits = F.interpolate(seg_logits, size=target_size, mode="bilinear", align_corners=False)

        seg_loss = self.dice_loss(seg_logits, targets) + self._cross_entropy_seg(seg_logits, targets)
        if edge_logits is None or self.lambda_edge <= 0:
            return seg_loss

        edge_targets = build_edge_targets(
            targets,
            method=self.edge_method,
            sobel_threshold=self.sobel_threshold,
        )
        if edge_logits.shape[-2:] != target_size:
            edge_logits = F.interpolate(edge_logits, size=target_size, mode="bilinear", align_corners=False)
        boundary_loss = self.bce_edge(edge_logits, edge_targets)
        return seg_loss + (self.lambda_edge * boundary_loss)
