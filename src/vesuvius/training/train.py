# src/vesuvius/training/train.py
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from vesuvius.data.vesuvius_train import VesuviusTrainDataset
from vesuvius.data.subset_with_transform import SubsetWithTransform

def make_dataloaders(
    data_root: Path,
    batch_size: int,
    val_fraction: float = 0.15,
    seed: int = 0,
    train_transform=None,
    val_transform=None,
):
    dataset = VesuviusTrainDataset(data_root, transform=None)  

    n_total = len(dataset)
    n_val = max(1, int(val_fraction * n_total)) # fraction of data for val
    n_train = n_total - n_val # remaining data for train

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=g)

    # Apply different transforms per split (assignment style)
    train_ds = SubsetWithTransform(train_subset, transform=train_transform)
    val_ds = SubsetWithTransform(val_subset, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, train_ds, val_ds