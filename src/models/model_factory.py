from .unet import AttEfficientUNetB0, AttEfficientUNetB4, SpaceNativeGhostNet

def get_model(model_name, n_classes=1, pretrained=True):
    """
    Girilen model ismine göre ilgili mimariyi döndürür.
    """
    model_name = model_name.lower()
    
    if model_name == "b0":
        return AttEfficientUNetB0(n_classes=n_classes, pretrained=pretrained)
    
    elif model_name == "b4":
        return AttEfficientUNetB4(n_classes=n_classes, pretrained=pretrained)
    
    elif model_name == "ghostnet":
        return SpaceNativeGhostNet(n_classes=n_classes, pretrained=pretrained)
    
    else:
        raise ValueError(f"HATA: '{model_name}' isminde bir model tanımlı değil. (b0, b4, ghostnet) deneyin.")
