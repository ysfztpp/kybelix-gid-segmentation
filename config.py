from __future__ import annotations

import os
import shutil
import zipfile
import torch


def _dir_has_data(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "unmasked"))
        and os.path.isdir(os.path.join(path, "masked"))
    )


def _find_data_root(search_root: str) -> str | None:
    if _dir_has_data(search_root):
        return search_root

    if not os.path.isdir(search_root):
        return None

    for name in sorted(os.listdir(search_root)):
        candidate = os.path.join(search_root, name)
        if _dir_has_data(candidate):
            return candidate

    return None


def _copy_zip_if_needed(src_zip: str, dst_zip: str) -> None:
    dst_dir = os.path.dirname(dst_zip)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    src_size = os.path.getsize(src_zip)
    dst_size = os.path.getsize(dst_zip) if os.path.isfile(dst_zip) else -1
    if src_size != dst_size:
        shutil.copy2(src_zip, dst_zip)


def _extract_zip(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _normalize_data_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or "drive_stream").strip().lower()
    aliases = {
        "drive": "drive_stream",
        "stream": "drive_stream",
        "zip": "zip_local_cache",
        "local_zip": "zip_local_cache",
    }
    return aliases.get(mode, mode)


def _resolve_base_path(shortcut_name: str, data_mode: str, data_root_override: str | None) -> str:
    if data_root_override:
        return data_root_override

    drive_path = f"/content/drive/MyDrive/{shortcut_name}"
    local_path = f"./{shortcut_name}"

    if data_mode == "drive_stream":
        # Eski davranış: Colab'de Drive klasöründen birebir oku.
        if os.path.exists("/content/drive"):
            return drive_path
        return local_path

    if data_mode == "zip_local_cache":
        # Colab: Drive'daki zip'i /content'e kopyala ve aç.
        if os.path.exists("/content/drive/MyDrive"):
            local_cache_root = f"/content/{shortcut_name}"
            local_zip_path = f"/content/{shortcut_name}.zip"
            drive_zip_path = f"/content/drive/MyDrive/{shortcut_name}.zip"

            cached_data_root = _find_data_root(local_cache_root)
            if cached_data_root:
                return cached_data_root

            if os.path.isfile(drive_zip_path):
                _copy_zip_if_needed(drive_zip_path, local_zip_path)
                _extract_zip(local_zip_path, "/content")
                extracted_data_root = _find_data_root(local_cache_root) or _find_data_root("/content")
                if extracted_data_root:
                    return extracted_data_root

            # Zip yoksa otomatik olarak eski Drive-stream yöntemine geri dön.
            if _dir_has_data(drive_path):
                return drive_path
            return local_cache_root

        # Lokal: ./phase1data.zip varsa aç ve kullan.
        local_zip_path = f"./{shortcut_name}.zip"
        local_data_root = _find_data_root(local_path)
        if local_data_root:
            return local_data_root

        if os.path.isfile(local_zip_path):
            _extract_zip(local_zip_path, ".")
            extracted_data_root = _find_data_root(local_path) or _find_data_root(".")
            if extracted_data_root:
                return extracted_data_root

        return local_path

    raise ValueError(
        "Unsupported data mode. Use one of: drive_stream, zip_local_cache "
        f"(got: {data_mode})"
    )


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
    PROGRESS_BAR = False  # Colab !python çıktısında satır spam'ini engeller
    LOG_INTERVAL = 100  # progress bar kapalıyken kaç adımda bir log basılsın
    METRIC_THRESHOLD = 0.5  # Binary metriklerde (IoU/mIoU/Kappa/OA) eşik değeri

    # --- Validation Raporlama ---
    # False: mevcut hızlı akış (rapor üretmez)
    # True: her epoch için detaylı validation raporu üretir
    ENABLE_VALIDATION_REPORTING = False
    REPORTING_TOP_K = 10  # En kötü kaç örnek detaylı kaydedilsin
    REPORTING_STATS_DIR_NAME = "stats"  # results/<run_name>/stats/epoch_XXX

    # --- Veri Seti Ayarları ---
    TARGET_COLOR = [0, 255, 0]  # GID Dataset yeşil alanlar
    USE_AUGMENTATION = True
    AUGMENTATION_PRESET = "strong"  # "none", "light", "strong"

    # --- Yol Ayarları (Colab & Local Uyumu) ---
    SHORTCUT_NAME = os.environ.get("KYBELIX_SHORTCUT_NAME", "phase1data")
    DATA_SOURCE_MODE = _normalize_data_mode(os.environ.get("KYBELIX_DATA_SOURCE_MODE", "drive_stream"))
    DATA_ROOT_OVERRIDE = os.environ.get("KYBELIX_DATA_ROOT")
    BASE_PATH = _resolve_base_path(
        shortcut_name=SHORTCUT_NAME,
        data_mode=DATA_SOURCE_MODE,
        data_root_override=DATA_ROOT_OVERRIDE,
    )

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
    EARLY_STOPPING_MONITOR = "val_iou"  # val_iou, val_miou, val_kappa, val_oa, val_loss

    # --- LR Scheduler ---
    SCHEDULER = "plateau"  # None veya "plateau"
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6


# Klasörlerin varlığından emin olalım
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.RESULT_DIR, exist_ok=True)
