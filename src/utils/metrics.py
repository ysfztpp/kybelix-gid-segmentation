import torch

from .model_outputs import get_segmentation_logits


def _prepare_binary_predictions(outputs, labels, threshold=0.5):
    outputs = get_segmentation_logits(outputs)
    preds = (torch.sigmoid(outputs) > threshold).float()
    labels = labels.float()
    if preds.shape != labels.shape:
        labels = labels.view_as(preds)
    return preds, labels


def get_iou_score(outputs, labels, threshold=0.5):
    """
    Cavit'in skor dosyalarındaki IoU mantığının PyTorch tensör versiyonu.
    Hem batch bazlı hem de tekli görüntülerde çalışır.
    """
    with torch.no_grad():
        preds, labels = _prepare_binary_predictions(outputs, labels, threshold=threshold)

        # 3. Kesişim (Intersection) ve Birleşim (Union) hesapla
        intersection = (preds * labels).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection
        
        # 4. Sıfıra bölünme hatasını engelle (Eğer her iki maske de boşsa IoU 1.0 kabul edilir)
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        # Batch ortalamasını döndür
        return iou.mean().item()


def get_confusion_counts(outputs, labels, threshold=0.5):
    with torch.no_grad():
        preds, labels = _prepare_binary_predictions(outputs, labels, threshold=threshold)
        tp = (preds * labels).sum().item()
        fp = (preds * (1.0 - labels)).sum().item()
        tn = ((1.0 - preds) * (1.0 - labels)).sum().item()
        fn = ((1.0 - preds) * labels).sum().item()
        return tp, fp, tn, fn


def get_confusion_counts_from_probs(probs, labels, threshold=0.5):
    with torch.no_grad():
        preds = (probs > threshold).float()
        labels = labels.float()
        if preds.shape != labels.shape:
            labels = labels.view_as(preds)
        tp = (preds * labels).sum().item()
        fp = (preds * (1.0 - labels)).sum().item()
        tn = ((1.0 - preds) * (1.0 - labels)).sum().item()
        fn = ((1.0 - preds) * labels).sum().item()
        return tp, fp, tn, fn


def get_binary_metrics_from_confusion(tp, fp, tn, fn):
    tp = float(tp)
    fp = float(fp)
    tn = float(tn)
    fn = float(fn)
    total = tp + fp + tn + fn

    if total <= 0:
        return {
            "iou_fg": 0.0,
            "iou_bg": 0.0,
            "miou": 0.0,
            "oa": 0.0,
            "kappa": 0.0,
        }

    fg_denom = tp + fp + fn
    bg_denom = tn + fp + fn
    iou_fg = tp / fg_denom if fg_denom > 0 else 1.0
    iou_bg = tn / bg_denom if bg_denom > 0 else 1.0
    miou = 0.5 * (iou_fg + iou_bg)

    oa = (tp + tn) / total
    pred_pos = tp + fp
    pred_neg = tn + fn
    gt_pos = tp + fn
    gt_neg = tn + fp
    pe = ((pred_pos * gt_pos) + (pred_neg * gt_neg)) / (total * total)
    kappa_denom = 1.0 - pe
    kappa = (oa - pe) / kappa_denom if abs(kappa_denom) > 1e-12 else 0.0

    return {
        "iou_fg": iou_fg,
        "iou_bg": iou_bg,
        "miou": miou,
        "oa": oa,
        "kappa": kappa,
    }


def get_dice_score(outputs, labels, threshold=0.5):
    """
    Ekstra: Dice katsayısı (F1-score) ölçümü. 
    Lider sorarsa 'Bunu da ekledim' dersin, puan kazandırır.
    """
    with torch.no_grad():
        preds, labels = _prepare_binary_predictions(outputs, labels, threshold=threshold)
            
        intersection = (preds * labels).sum(dim=(1, 2, 3))
        dice = (2. * intersection + 1e-7) / (preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) + 1e-7)
        return dice.mean().item()
