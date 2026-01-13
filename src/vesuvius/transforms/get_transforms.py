# src/vesuvius/transforms/get_transforms.py
from __future__ import annotations
from vesuvius.transforms import Compose, NormalizeVolume, CenterCrop3D, RandomCrop3D

def get_transformations(
        crop_size=(64, 256, 256),
        use_random_crop_for_train=True,
        mean=None,
        std=None,
):
    """
    Returns:
      main_transform: deterministic (val/test)
      train_transform: includes random crop (train)
    """
    main_transform = Compose([
        CenterCrop3D(crop_size),
        NormalizeVolume(mean=mean, std=std),
    ])

    if use_random_crop_for_train:
        train_transform = Compose([
            RandomCrop3D(crop_size),
            NormalizeVolume(mean=mean, std=std),
        ])
    else:
        train_transform = main_transform

    return main_transform, train_transform