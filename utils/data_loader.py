from torch.utils.data import DataLoader, Dataset
from typing import Tuple

__all__ = [
    "ReviewDataset",
    "get_data_loaders",
]


class ReviewDataset(Dataset):
    """Simple torch Dataset wrapping the preprocessed HuggingFace split."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["user_ids"])

    def __getitem__(self, idx):
        return {
            "user_ids": self.data["user_ids"][idx],
            "product_ids": self.data["product_ids"][idx],
            "ratings": self.data["ratings"][idx],
            "product_titles": self.data["product_titles"][idx],
        }


def get_data_loaders(train_data, test_data, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return PyTorch DataLoaders for train & validation."""
    train_loader = DataLoader(ReviewDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ReviewDataset(test_data), batch_size=batch_size)
    return train_loader, val_loader 