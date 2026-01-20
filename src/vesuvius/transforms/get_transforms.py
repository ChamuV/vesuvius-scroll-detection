# src/vesuvius/transforms/get_transforms.py
# src/vesuvius/transforms/get_transforms.py
from __future__ import annotations

from vesuvius.transforms import (
    Compose,
    NormalizeVolume,
    CenterCrop3D,
    RandomCrop3D,
    RandomFlip3D,
    RandomIntensityAffine,
    RandomGaussianNoise,
)


def get_transformations(
    crop_size=(64, 256, 256),
    use_random_crop_for_train=True,
    mean=None,
    std=None,
    *,
    # augmentation toggles / strengths
    use_flips: bool = True,
    use_intensity_jitter: bool = True,
    use_gaussian_noise: bool = True,
):
    """
    Returns:
      main_transform : deterministic (val/test)
      train_transform: stochastic (train)

    All transforms operate on dicts with keys:
      - "image": (1, D, H, W)
      - "mask" : (1, D, H, W) (optional)
    """
    main_transform = Compose([
        CenterCrop3D(crop_size),
        NormalizeVolume(mean=mean, std=std),
    ])

    if use_random_crop_for_train:
        train_transforms = [RandomCrop3D(crop_size)]
    else:
        train_transforms = [CenterCrop3D(crop_size)]

    if use_flips:
        train_transforms.append(
            RandomFlip3D(p=0.5, axes=("H", "W"))
        )

    if use_intensity_jitter:
        train_transforms.append(
            RandomIntensityAffine(
                scale_range=(0.9, 1.1),
                shift_range=(-0.1, 0.1),
                p=0.5,
            )
        )

    if use_gaussian_noise:
        train_transforms.append(
            RandomGaussianNoise(
                std_range=(0.0, 0.03),
                p=0.5,
            )
        )

    train_transforms.append(
        NormalizeVolume(mean=mean, std=std)
    )

    train_transform = Compose(train_transforms)

    return main_transform, train_transform