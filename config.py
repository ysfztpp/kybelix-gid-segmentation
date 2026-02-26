import os
import torch

class Config:
    # --- Genel Ayarlar ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # --- Model Seçimi ---
    # Senin yeni modelin factory'de 'b0_custom' olarak kayıtlı olmalı
    MODEL_NAME = "b0_custom" 
    NUM_CLASSES = 1
    PRETRAINED = True

    # --- Hiperparametreler (T4 & 27k Veri İçin Sıkı Optimizasyon) ---
    BATCH_SIZE = 8          # T4 (16GB) için 512x512'de güvenli sınır
    ACCUMULATION_STEPS = 4  # 8 x 4 = 32 Sanal Batch Size (Stabilite sağlar)
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512  
    NUM_WORKERS = 2         # CPU'da paralel veri hazırlama
    PIN_MEMORY = True       # GPU'ya veri transferini hızlandırır

    # --- Kademeli Eğitim (Freezing) ---
    FREEZE_ENCODER = True   # Başlangıçta encoder dondurulsun mu?
    UNFREEZE_EPOCH = 3      # 3. epoch'ta tüm katmanları aç

    # --- Veri Seti Ayarları ---
    TARGET_COLOR = [0, 255, 0]  # GID Dataset yeşil alanlar
    USE_AUGMENTATION = True
    AUGMENTATION_PRESET = "strong"

    # --- Yol Ayarları (Değiştirilmedi) ---
    SHORTCUT_NAME = "phase1data"
    if os.path.exists("/content/drive"):
        BASE_PATH = f"/content/drive/MyDrive/{SHORTCUT_NAME}"
    else:
        BASE_PATH = f"./{SHORTCUT_NAME}"

    TRAIN_IMG_DIR = os.path.join(BASE_PATH, "unmasked", "train")
    TRAIN_MSK_DIR = os.path.join(BASE_PATH, "masked", "train")
    VAL_IMG_DIR = os.path.join(BASE_PATH, "unmasked", "val")
    VAL_MSK_DIR = os.path.join(BASE_PATH, "masked", "val")

    # --- Kayıt ve Sonuçlar ---
    SAVE_RUNS_TO_DRIVE = True
    RUNS_DIR_NAME = "kybelix_runs"
    SAVE_EPOCHS_LOCALLY = True
    LOCAL_RUNS_ROOT = "."

    if SAVE_RUNS_TO_DRIVE and os.path.exists("/content/drive"):
        RUNS_ROOT = f"/content/drive/MyDrive/{RUNS_DIR_NAME}"
    else:
        RUNS_ROOT = "."

    CHECKPOINT_DIR = os.path.join(RUNS_ROOT, "checkpoints")
    RESULT_DIR = os.path.join(RUNS_ROOT, "results")
    LOCAL_CHECKPOINT_DIR = os.path.join(LOCAL_RUNS_ROOT, "checkpoints")
    RESUME_PATH = None
    # --- Early Stopping & Scheduler ---
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MONITOR = "val_iou"
    SCHEDULER = "plateau"
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.RESULT_DIR, exist_ok=True)
