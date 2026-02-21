import os
import torch


class Config:
    # --- Genel Ayarlar ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # --- Model Seçimi ---
    # Seçenekler: "b0", "b4", "ghostnet"
    MODEL_NAME = "b4"
    NUM_CLASSES = 1
    PRETRAINED = True

    # --- Hiperparametreler ---
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512  # Giriş görüntü boyutu
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # --- Veri Seti Ayarları ---
    TARGET_COLOR = [0, 255, 0]  # GID Dataset yeşil alanlar

    # --- Yol Ayarları (Colab & Local Uyumu) ---
    SHORTCUT_NAME = "phase1data"

    if os.path.exists("/content/drive"):
        BASE_PATH = f"/content/drive/MyDrive/{SHORTCUT_NAME}"
    else:
        BASE_PATH = f"./{SHORTCUT_NAME}"  # Lokal test için

    # Alt klasör yolları
    TRAIN_IMG_DIR = os.path.join(BASE_PATH, "unmasked", "train")
    TRAIN_MSK_DIR = os.path.join(BASE_PATH, "masked", "train")
    VAL_IMG_DIR = os.path.join(BASE_PATH, "unmasked", "val")
    VAL_MSK_DIR = os.path.join(BASE_PATH, "masked", "val")

    # --- Kayıt ve Sonuçlar ---
    CHECKPOINT_DIR = "checkpoints"
    RESULT_DIR = "results"


# Klasörlerin varlığından emin olalım
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.RESULT_DIR, exist_ok=True)
