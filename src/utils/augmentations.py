import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_augmentations(config, is_train: bool = True):
    if not is_train:
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                ToTensorV2(),
            ]
        )

    preset = getattr(config, "AUGMENTATION_PRESET", "strong")
    if preset == "none":
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                ToTensorV2(),
            ]
        )

    if preset == "light":
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.4),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.08, rotate_limit=15, p=0.6),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.HueSaturationValue(p=0.3),
            ToTensorV2(),
        ]
    )
