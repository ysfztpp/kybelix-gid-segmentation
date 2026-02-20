import torch
import os

class Config:
    # --- Genel Ayarlar ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # --- Model Seçimi ---
    # Seçenekler: "b0", "b4", "ghostnet"
    # Factory sayesinde buradan ismi değiştirmek yeterli
    MODEL_NAME = "b4" 
    NUM_CLASSES = 1
    
    # --- Hiperparametreler ---
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512  # Giriş görüntü boyutu
    
    # --- Veri Seti Ayarları ---
    TARGET_COLOR = [0, 255, 0] # GID Dataset yeşil alanlar
    
    # --- Yol Ayarları (Colab & Local Uyumu) ---
    # Drive kısayol ismi veya lokal klasör ismi
    SHORTCUT_NAME = 'Dataset_Final'
    
    if os.path.exists('/content/drive'):
        BASE_PATH = f'/content/drive/MyDrive/{SHORTCUT_NAME}'
    else:
        BASE_PATH = f'./{SHORTCUT_NAME}' # Lokal test için

    # Alt klasör yollarını otomatik oluşturuyoruz
    TRAIN_IMG_DIR = os.path.join(BASE_PATH, "Normal", "train")
    TRAIN_MSK_DIR = os.path.join(BASE_PATH, "Mask", "train")
    VAL_IMG_DIR = os.path.join(BASE_PATH, "Normal", "val")
    VAL_MSK_DIR = os.path.join(BASE_PATH, "Mask", "val")
    
    # --- Kayıt ve Sonuçlar ---
    CHECKPOINT_DIR = "checkpoints"
    RESULT_DIR = "results"

# Klasörlerin varlığından emin olalım
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.RESULT_DIR, exist_ok=True)
