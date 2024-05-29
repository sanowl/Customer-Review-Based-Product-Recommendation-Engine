import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("amazon_us_reviews", "Video_Games_v1_00")

# Preprocess the dataset
def preprocess_data(examples):
    user_ids = examples["reviewer_id"]
    product_ids = examples["product_id"]
    ratings = examples["star_rating"]
    product_titles = examples["product_title"]
    return {"user_ids": user_ids, "product_ids": product_ids, "ratings": ratings, "product_titles": product_titles}

preprocessed_dataset = dataset.map(preprocess_data, remove_columns=dataset["train"].column_names)

# Label encoding for users and products
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

preprocessed_dataset = preprocessed_dataset.map(lambda x: {
    "user_ids": user_encoder.fit_transform(x["user_ids"]),
    "product_ids": product_encoder.fit_transform(x["product_ids"]),
    "ratings": x["ratings"],
    "product_titles": x["product_titles"]
})

# Split the dataset into train and test sets
train_dataset = preprocessed_dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = train_dataset["train"]
test_data = train_dataset["test"]

# Define a custom dataset class
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["user_ids"])

    def __getitem__(self, idx):
        return {
            "user_ids": self.data["user_ids"][idx],
            "product_ids": self.data["product_ids"][idx],
            "ratings": self.data["ratings"][idx],
            "product_titles": self.data["product_titles"][idx]
        }

# Initialize data loaders
train_loader = DataLoader(ReviewDataset(train_data), batch_size=128, shuffle=True)
test_loader = DataLoader(ReviewDataset(test_data), batch_size=128)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Define the recommendation model
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.product_embeddings = nn.Embedding(num_products, embedding_dim)
        self.bert_model = bert_model
        self.fc1 = nn.Linear(embedding_dim + bert_model.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, user_ids, product_ids, product_titles):
        user_emb = self.user_embeddings(user_ids)
        product_emb = self.product_embeddings(product_ids)
        
        product_features = self.bert_model(**product_titles).last_hidden_state[:, 0, :]
        combined_emb = torch.cat((user_emb, product_features), dim=1)
        
        x = nn.ReLU()(self.fc1(combined_emb))
        output = self.fc2(x)
        return output

# Set up the model and training parameters
num_users = len(user_encoder.classes_)
num_products = len(product_encoder.classes_)
embedding_dim = 64
learning_rate = 0.001
num_epochs = 10

# Determine the device to use (CUDA, MPS for M1 Macs, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = RecommendationModel(num_users, num_products, embedding_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        user_ids = batch["user_ids"].to(device)
        product_ids = batch["product_ids"].to(device)
        ratings = batch["ratings"].to(device)
        
        product_titles = tokenizer(batch["product_titles"], padding=True, truncation=True, return_tensors="pt").to(device)
        
        optimizer.zero_grad()
        predicted_ratings = model(user_ids, product_ids, product_titles).squeeze()
        loss = criterion(predicted_ratings, ratings.float())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            user_ids = batch["user_ids"].to(device)
            product_ids = batch["product_ids"].to(device)
            ratings = batch["ratings"].to(device)
            
            product_titles = tokenizer(batch["product_titles"], padding=True, truncation=True, return_tensors="pt").to(device)
            
            predicted_ratings = model(user_ids, product_ids, product_titles).squeeze()
            loss = criterion(predicted_ratings, ratings.float())
            epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, test_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "recommendation_model.pth")

# Generate recommendations
def recommend_products(user_id, top_k=5):
    user_id_enc = torch.tensor([user_encoder.transform([user_id])[0]] * num_products).to(device)
    product_ids = torch.arange(num_products).to(device)
    
    unique_product_titles = preprocessed_dataset["product_titles"].unique()
    product_titles = tokenizer(unique_product_titles, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        predicted_ratings = model(user_id_enc, product_ids, product_titles).squeeze()
        top_product_indices = torch.topk(predicted_ratings, top_k).indices
        recommended_products = unique_product_titles[top_product_indices.cpu().numpy()]
    
    return recommended_products

# Example usage
user_id = "A2VNYWOPJ6U2EV"
recommended_products = recommend_products(user_id)
print(f"Recommended products for user {user_id}:")
for product in recommended_products:
    print(product)
