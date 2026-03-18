import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_outputs import split_model_outputs


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    labels_sorted = labels[perm]
    grad = _lovasz_grad(labels_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels, per_image=True):
    if per_image:
        losses = []
        for logit, label in zip(logits, labels):
            losses.append(_lovasz_hinge_flat(logit.view(-1), label.view(-1)))
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0
    return _lovasz_hinge_flat(logits.view(-1), labels.view(-1))


def _flatten_probas(probas, labels):
    if probas.dim() == 3:
        probas = probas.unsqueeze(1)
    c = probas.shape[1]
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.view(-1)
    return probas, labels


def _lovasz_softmax_flat(probas, labels):
    if probas.numel() == 0:
        return probas.sum() * 0.0
    num_classes = probas.shape[1]
    losses = []
    for class_id in range(num_classes):
        foreground = (labels == class_id).float()
        if foreground.sum() == 0:
            continue
        class_pred = probas[:, class_id]
        errors = (foreground - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        foreground_sorted = foreground[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(probas, labels, per_image=False):
    if per_image:
        losses = []
        for prob, label in zip(probas, labels):
            prob_flat, label_flat = _flatten_probas(prob.unsqueeze(0), label.unsqueeze(0))
            losses.append(_lovasz_softmax_flat(prob_flat, label_flat))
        return torch.stack(losses).mean() if losses else probas.sum() * 0.0
    prob_flat, label_flat = _flatten_probas(probas, labels)
    return _lovasz_softmax_flat(prob_flat, label_flat)

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


class FocalLoss(nn.Module):
    """
    Focal Loss for binary or multi-class segmentation.
    alpha:
      - binary: float (pos weight) or [alpha_neg, alpha_pos]
      - multi-class: list/tuple/tensor of per-class weights
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def _resolve_alpha(self, device, dtype):
        if self.alpha is None:
            return None
        if torch.is_tensor(self.alpha):
            return self.alpha.to(device=device, dtype=dtype)
        if isinstance(self.alpha, (list, tuple)):
            return torch.tensor(self.alpha, device=device, dtype=dtype)
        return torch.tensor(float(self.alpha), device=device, dtype=dtype)

    def _reduce(self, loss):
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def _binary_focal(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        if logits.shape != targets.shape:
            targets = targets.view_as(logits)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_term = (1.0 - pt).pow(self.gamma)

        alpha = self._resolve_alpha(logits.device, logits.dtype)
        if alpha is None:
            alpha_t = 1.0
        else:
            if alpha.numel() == 1:
                alpha_pos = alpha
                alpha_neg = 1.0 - alpha
            elif alpha.numel() == 2:
                alpha_neg = alpha[0]
                alpha_pos = alpha[1]
            else:
                raise ValueError("Binary focal alpha must be scalar or length 2.")
            alpha_t = alpha_pos * targets + alpha_neg * (1.0 - targets)

        loss = alpha_t * focal_term * bce
        return self._reduce(loss)

    def _multiclass_focal(self, logits, targets):
        if targets.dim() == 4:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)
            else:
                targets = targets.argmax(dim=1)
        targets = targets.long()

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        logpt = log_probs.gather(1, targets.unsqueeze(1))
        pt = probs.gather(1, targets.unsqueeze(1))
        focal_term = (1.0 - pt).pow(self.gamma)

        alpha = self._resolve_alpha(logits.device, logits.dtype)
        if alpha is None:
            alpha_t = 1.0
        else:
            if alpha.numel() == 1:
                alpha_t = alpha
            else:
                if alpha.numel() != logits.shape[1]:
                    raise ValueError(
                        f"Multi-class focal alpha must have {logits.shape[1]} values, got {alpha.numel()}."
                    )
                alpha_t = alpha[targets].unsqueeze(1)

        loss = -alpha_t * focal_term * logpt
        return self._reduce(loss.squeeze(1))

    def forward(self, logits, targets):
        if logits.shape[1] == 1:
            return self._binary_focal(logits, targets)
        return self._multiclass_focal(logits, targets)


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
    Total Loss = Dice(seg) + SegLoss(seg) + lambda * BCE(edge)
    SegLoss = CrossEntropy/BCE or Focal Loss (optional)
    """

    def __init__(
        self,
        lambda_edge=0.4,
        edge_method="sobel",
        sobel_threshold=0.1,
        enable_lovasz=False,
        lovasz_weight=0.3,
        lovasz_per_image=True,
        pos_weight=None,
        class_weight=None,
        use_focal=False,
        focal_gamma=2.0,
        focal_alpha=None,
        focal_reduction="mean",
    ):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.edge_method = edge_method
        self.sobel_threshold = sobel_threshold
        self.enable_lovasz = enable_lovasz
        self.lovasz_weight = lovasz_weight
        self.lovasz_per_image = lovasz_per_image

        self.use_focal = bool(use_focal)
        self.dice_loss = DiceLoss()
        self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        self.ce_seg = nn.CrossEntropyLoss(weight=class_weight) if class_weight is not None else nn.CrossEntropyLoss()
        self.focal_loss = (
            FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction=focal_reduction)
            if self.use_focal
            else None
        )
        self.bce_edge = nn.BCEWithLogitsLoss()

    def _cross_entropy_seg(self, seg_logits, targets):
        if self.use_focal:
            return self.focal_loss(seg_logits, targets)

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

        if self.enable_lovasz and self.lovasz_weight > 0:
            if seg_logits.shape[1] == 1:
                binary_targets = targets
                if binary_targets.dim() == 4 and binary_targets.shape[1] == 1:
                    binary_targets = binary_targets[:, 0]
                binary_targets = binary_targets.float()
                lovasz = lovasz_hinge(
                    logits=seg_logits[:, 0],
                    labels=binary_targets,
                    per_image=self.lovasz_per_image,
                )
            else:
                if targets.dim() == 4:
                    if targets.shape[1] == 1:
                        class_targets = targets.squeeze(1).long()
                    else:
                        class_targets = targets.argmax(dim=1).long()
                else:
                    class_targets = targets.long()
                probs = torch.softmax(seg_logits, dim=1)
                lovasz = lovasz_softmax(
                    probas=probs,
                    labels=class_targets,
                    per_image=self.lovasz_per_image,
                )
            seg_loss = seg_loss + (self.lovasz_weight * lovasz)

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
