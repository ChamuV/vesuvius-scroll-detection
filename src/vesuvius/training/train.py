# src/vesuvius/training/train.py
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from vesuvius.data.vesuvius_train import VesuviusTrainDataset

def make_dataloaders(
        data_root: Path,
        batch_size: int,
        val_fraction: float = 0.15,
):
    # Pass data through dataloader
    dataset = VesuviusTrainDataset(data_root)

    n_total = len(dataset) # length of dataset
    n_val = max(1, int(val_fraction * n_total)) # fraction of dataset allocated for valuation
    n_train = n_total - n_val # fraction of dataset allocated for traininig

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader( 
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    return train_loader, val_loader, train_ds, val_ds