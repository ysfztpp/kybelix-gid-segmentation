import os
import torch


class Config:
    # --- Genel Ayarlar ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # --- Model Seçimi ---
    # Seçenekler: "b0", "b4", "ghostnet", "segformer_b4"
    MODEL_NAME = "segformer_b4"
    NUM_CLASSES = 1
    PRETRAINED = True

    # --- Boundary-aware yardımcı kayıp ---
    # Total Loss = Dice(seg) + CrossEntropy(seg) + lambda * BCE(edge)
    EDGE_LOSS_WEIGHT = 0.4  # 0.3 - 0.5 arası önerilir
    EDGE_TARGET_METHOD = "sobel"  # "sobel" veya "morph"
    EDGE_SOBEL_THRESHOLD = 0.1

    # --- Hiperparametreler (T4 için optimize edildi) ---
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 1  # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512  # Giriş görüntü boyutu
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # --- Veri Seti Ayarları ---
    TARGET_COLOR = [0, 255, 0]  # GID Dataset yeşil alanlar
    USE_AUGMENTATION = True
    AUGMENTATION_PRESET = "strong"  # "none", "light", "strong"

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

    # --- Resume Ayarları ---
    # Örnek: "checkpoints/<run_name>/segformer_b4_last.pth"
    RESUME_PATH = None
    # True ise optimizer state yüklenmez
    RESET_OPTIMIZER = False

    # --- Early Stopping ---
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-4
    EARLY_STOPPING_MONITOR = "val_iou"  # "val_iou" veya "val_loss"

    # --- LR Scheduler ---
    SCHEDULER = "plateau"  # None veya "plateau"
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6


# Klasörlerin varlığından emin olalım
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.RESULT_DIR, exist_ok=True)
