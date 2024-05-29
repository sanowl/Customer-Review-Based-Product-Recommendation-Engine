import torch

# Dataset configuration
DATASET_NAME = "amazon_us_reviews"
SUBSET_NAME = "Video_Games_v1_00"

# Model configuration
EMBEDDING_DIM = 64
BERT_MODEL_NAME = "bert-base-uncased"

# Training configuration
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 128

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")