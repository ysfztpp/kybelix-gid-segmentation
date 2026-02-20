import torch
import torch.nn as nn

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
