# config.py
import os

# Google Drive Ayarları
SHORTCUT_NAME = 'Dataset_Final'
# Colab'de '/content/drive/MyDrive/...' olur, lokalde test ederken hata vermemesi için basit bir kontrol:
if os.path.exists('/content/drive'):
    BASE_PATH = f'/content/drive/MyDrive/{SHORTCUT_NAME}/'
else:
    BASE_PATH = f'./{SHORTCUT_NAME}/' # Lokal test için

# Model Parametreleri
TARGET_COLOR_RGB = [0, 255, 0]
BATCH_SIZE = 4
LIMIT = 32