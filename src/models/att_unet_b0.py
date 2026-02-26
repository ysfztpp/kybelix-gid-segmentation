import torch
import torch.nn as nn
import timm

# ==========================================
# 1. YARDIMCI MODÜL: DEPTHWISE SEPARABLE CONV
# ==========================================
# Standart Conv yerine bunu kullanarak RAM kullanımını ve FLOPs değerini düşürüyoruz.
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=kernel_size, padding=padding, groups=in_c, bias=bias)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU6(inplace=True) # İnovasyon: ReLU6 kullanımı

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# ==========================================
# 2. YARDIMCI MODÜL: ATTENTION GATE
# ==========================================
# Skip-connection'dan gelen veriyi süzen "Süzgeç" yapısı.
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
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
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ==========================================
# 3. ANA MODEL: ATTENTION EFFICIENT-UNET B0
# ==========================================
class AttEfficientUNetB0(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        
        # --- ENCODER: EfficientNet-B0 ---
        # features_only=True ile ara katmanları (skip connections) alıyoruz.
        self.encoder = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)
        
        # --- İNOVASYON: Encoder Aktivasyonlarını ReLU6 ile Değiştirme ---
        # Swish (SiLU) yerine Edge-TPU dostu ReLU6'ya geçiyoruz.
        self._replace_swish_with_relu6(self.encoder)
        
        # B0 Feature Channels (Encoder'dan gelen): [16, 24, 40, 112, 320]
        ch = [16, 24, 40, 112, 320]

        # --- DECODER BLOKLARI ---
        # Her blokta Depthwise Separable Conv kullanıyoruz (RAM tasarrufu)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(f_g=ch[3], f_l=ch[3], f_int=ch[3]//2)
        self.dec1 = DepthwiseSeparableConv(ch[4] + ch[3], ch[3]) # 320 + 112 -> 112

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(f_g=ch[2], f_l=ch[2], f_int=ch[2]//2)
        self.dec2 = DepthwiseSeparableConv(ch[3] + ch[2], ch[2]) # 112 + 40 -> 40

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(f_g=ch[1], f_l=ch[1], f_int=ch[1]//2)
        self.dec3 = DepthwiseSeparableConv(ch[2] + ch[1], ch[1]) # 40 + 24 -> 24

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DepthwiseSeparableConv(ch[1] + ch[0], ch[0]) # 24 + 16 -> 16
        
        # Final katmanı (Sınıflandırma)
        self.final_conv = nn.Conv2d(ch[0], n_classes, kernel_size=1)

    def _replace_swish_with_relu6(self, model):
        """Encoder içindeki tüm Swish/SiLU katmanlarını ReLU6 ile takas eder."""
        for name, module in model.named_children():
            if isinstance(module, (nn.SiLU, nn.ReLU)):
                setattr(model, name, nn.ReLU6(inplace=True))
            else:
                self._replace_swish_with_relu6(module)

    def forward(self, x):
        # Encoder aşaması
        features = self.encoder(x)
        # B0 features listesi: [stride2, stride4, stride8, stride16, stride32]
        # skip_connections: e0, e1, e2, e3
        e0, e1, e2, e3, e4 = features 

        # Decoder aşaması 1
        d1 = self.up1(e4)
        a1 = self.att1(g=d1, x=e3)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.dec1(d1)

        # Decoder aşaması 2
        d2 = self.up2(d1)
        a2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.dec2(d2)

        # Decoder aşaması 3
        d3 = self.up3(d2)
        a3 = self.att3(g=d3, x=e1)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.dec3(d3)

        # Decoder aşaması 4 (Skip Connection: e0)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e0], dim=1)
        d4 = self.dec4(d4)

        # Final çıktı
        out = self.final_conv(d4)
        # Orijinal boyuta dönmek için (512x512)
        return nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear')
