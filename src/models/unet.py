import torch
import torch.nn as nn
import timm

# ==========================================
# ORTAK YARDIMCI MODÜLLER
# ==========================================

class AttentionBlock(nn.Module):
    """
    Skip-connection özelliklerini decoder gate (g) ile filtreleyen 
    Attention Gate modülü. (B0 ve B4 modelleri için ortak)
    """
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

def make_up_block(in_c, out_c):
    """Standart Upsampling + Conv bloğu"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

# ==========================================
# 1. MODEL: ATTENTION EFFICIENT-UNET (B0)
# ==========================================

class AttEfficientUNetB0(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)
        # B0 Feature Channels: [16, 24, 40, 112, 320]
        
        self.up1 = make_up_block(320, 112)
        self.att1 = AttentionBlock(f_g=112, f_l=112, f_int=56)
        
        self.up2 = make_up_block(112 * 2, 40)
        self.att2 = AttentionBlock(f_g=40, f_l=40, f_int=20)
        
        self.up3 = make_up_block(40 * 2, 24)
        self.att3 = AttentionBlock(f_g=24, f_l=24, f_int=12)
        
        self.up4 = make_up_block(24 * 2, 16)
        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        e0, e1, e2, e3 = features[1], features[2], features[3], features[4]
        
        d1 = self.up1(e3)
        e2_att = self.att1(g=d1, x=e2)
        d1 = torch.cat([d1, e2_att], dim=1)
        
        d2 = self.up2(d1)
        e1_att = self.att2(g=d2, x=e1)
        d2 = torch.cat([d2, e1_att], dim=1)
        
        d3 = self.up3(d2)
        e0_att = self.att3(g=d3, x=e0)
        d3 = torch.cat([d3, e0_att], dim=1)
        
        out = self.final_conv(self.up4(d3))
        return nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear')

# ==========================================
# 2. MODEL: ATTENTION EFFICIENT-UNET (B4)
# ==========================================

class AttEfficientUNetB4(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        ch = self.encoder.feature_info.channels() # [24, 32, 56, 160, 448]
        
        self.up1 = make_up_block(ch[4], ch[3])
        self.att1 = AttentionBlock(f_g=ch[3], f_l=ch[3], f_int=ch[3]//2)
        
        self.up2 = make_up_block(ch[3]*2, ch[2])
        self.att2 = AttentionBlock(f_g=ch[2], f_l=ch[2], f_int=ch[2]//2)
        
        self.up3 = make_up_block(ch[2]*2, ch[1])
        self.att3 = AttentionBlock(f_g=ch[1], f_l=ch[1], f_int=ch[1]//2)
        
        self.up4 = make_up_block(ch[1]*2, 24)
        self.final_conv = nn.Conv2d(24, n_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        e0, e1, e2, e3, e4 = features
        
        d1 = self.up1(e4)
        a1 = self.att1(g=d1, x=e3)
        d1 = torch.cat([d1, a1], dim=1)
        
        d2 = self.up2(d1)
        a2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, a2], dim=1)
        
        d3 = self.up3(d2)
        a3 = self.att3(g=d3, x=e1)
        d3 = torch.cat([d3, a3], dim=1)
        
        out = self.final_conv(self.up4(d3))
        return nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear')

# ==========================================
# 3. MODEL: SPACE-NATIVE GHOSTNET UNET
# ==========================================

class SpaceNativeGhostNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model('ghostnet_100', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        ch = self.encoder.feature_info.channels()
        
        self.up1 = make_up_block(ch[3], ch[2])
        self.up2 = make_up_block(ch[2] * 2, ch[1])
        self.up3 = make_up_block(ch[1] * 2, ch[0])    
        self.final_conv = nn.Conv2d(ch[0] * 2, n_classes, kernel_size=1)
    
    def forward(self, x):
        features = self.encoder(x)
        
        x_dec = self.up1(features[3])
        x_dec = torch.cat([x_dec, features[2]], dim=1)
        
        x_dec = self.up2(x_dec)
        x_dec = torch.cat([x_dec, features[1]], dim=1)
        
        x_dec = self.up3(x_dec)
        x_dec = torch.cat([x_dec, features[0]], dim=1)
        
        out = self.final_conv(x_dec)
        # Giriş boyutuna geri döndürme
        return nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear')
