from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "load_and_preprocess_dataset",
    "preprocess_data",
]


def preprocess_data(examples):
    """Extract only the columns we need and rename them."""
    return {
        "user_ids": examples["reviewer_id"],
        "product_ids": examples["product_id"],
        "ratings": examples["star_rating"],
        "product_titles": examples["product_title"],
    }


def _fit_label_encoders(dataset_split, user_encoder: LabelEncoder, product_encoder: LabelEncoder):
    """Fit encoders on a HuggingFace Dataset and return transformed features."""
    return dataset_split.map(
        lambda batch: {
            "user_ids": user_encoder.transform(batch["user_ids"]),
            "product_ids": product_encoder.transform(batch["product_ids"]),
            "ratings": batch["ratings"],
            "product_titles": batch["product_titles"],
        }
    )


def load_and_preprocess_dataset(dataset_name: str, subset_name: str, test_size: float = 0.2, seed: int = 42):
    """Download dataset, apply minimal cleaning, encode IDs, and create train/test splits.

    Returns:
        train_data, test_data, user_encoder, product_encoder, full_preprocessed_dataset
    """
    # Download
    raw_ds = load_dataset(dataset_name, subset_name)

    # Column selection & rename
    processed_ds = raw_ds.map(preprocess_data, remove_columns=raw_ds["train"].column_names)

    # Label encoders
    user_enc = LabelEncoder()
    product_enc = LabelEncoder()

    # We need to fit on full training split to avoid information leak from test
    user_enc.fit(processed_ds["train"]["user_ids"])
    product_enc.fit(processed_ds["train"]["product_ids"])

    encoded_ds = processed_ds.map(
        lambda batch: {
            "user_ids": user_enc.transform(batch["user_ids"]),
            "product_ids": product_enc.transform(batch["product_ids"]),
            "ratings": batch["ratings"],
            "product_titles": batch["product_titles"],
        }
    )

    # Train/val split
    split = encoded_ds["train"].train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"], user_enc, product_enc, encoded_ds 