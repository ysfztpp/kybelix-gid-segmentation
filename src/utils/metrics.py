import torch

from .model_outputs import get_segmentation_logits

def get_iou_score(outputs, labels, threshold=0.5):
    """
    Cavit'in skor dosyalarındaki IoU mantığının PyTorch tensör versiyonu.
    Hem batch bazlı hem de tekli görüntülerde çalışır.
    """
    with torch.no_grad():
        outputs = get_segmentation_logits(outputs)
        # 1. Logitleri olasılığa çevir (Sigmoid) ve binary (0-1) maske yap
        preds = (torch.sigmoid(outputs) > threshold).float()
        labels = labels.float()
        
        # 2. Boyutları eşitle (B, 1, H, W) formatında olduklarından emin oluyoruz
        if preds.shape != labels.shape:
            labels = labels.view_as(preds)

        # 3. Kesişim (Intersection) ve Birleşim (Union) hesapla
        intersection = (preds * labels).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection
        
        # 4. Sıfıra bölünme hatasını engelle (Eğer her iki maske de boşsa IoU 1.0 kabul edilir)
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        # Batch ortalamasını döndür
        return iou.mean().item()

def get_dice_score(outputs, labels, threshold=0.5):
    """
    Ekstra: Dice katsayısı (F1-score) ölçümü. 
    Lider sorarsa 'Bunu da ekledim' dersin, puan kazandırır.
    """
    with torch.no_grad():
        outputs = get_segmentation_logits(outputs)
        preds = (torch.sigmoid(outputs) > threshold).float()
        labels = labels.float()
        if preds.shape != labels.shape:
            labels = labels.view_as(preds)
            
        intersection = (preds * labels).sum(dim=(1, 2, 3))
        dice = (2. * intersection + 1e-7) / (preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) + 1e-7)
        return dice.mean().item()
