# Mevcut importlar (Eski yapıyı bozmuyoruz)
from .unet import AttEfficientUNetB0, AttEfficientUNetB4, SpaceNativeGhostNet

# YENİ EKLEME: Senin yazdığın modifiye edilmiş B0 modelini import ediyoruz
from .att_unet_b0 import AttEfficientUNetB0 as OptimizedAttEfficientUNetB0

def get_model(model_name, n_classes=1, pretrained=True):
    """
    Girilen model ismine göre ilgili mimariyi döndürür.
    """
    model_name = model_name.lower()
    
    # 1. ESKİ B0 (Orijinal yapı)
    if model_name == "b0":
        return AttEfficientUNetB0(n_classes=n_classes, pretrained=pretrained)
    
    # 2. SENİN YENİ B0 MODELİN (ReLU6 + Depthwise + Attention Gate)
    elif model_name == "b0_custom":
        return OptimizedAttEfficientUNetB0(n_classes=n_classes, pretrained=pretrained)
    
    # 3. B4 MODELİ
    elif model_name == "b4":
        return AttEfficientUNetB4(n_classes=n_classes, pretrained=pretrained)
    
    # 4. GHOSTNET
    elif model_name == "ghostnet":
        return SpaceNativeGhostNet(n_classes=n_classes, pretrained=pretrained)
    
    else:
        raise ValueError(f"HATA: '{model_name}' isminde bir model tanımlı değil. (b0, b0_custom, b4, ghostnet) deneyin.")
