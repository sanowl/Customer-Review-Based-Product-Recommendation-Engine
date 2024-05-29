import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from models.recommendation_model import RecommendationModel
from utils.data_loader import get_data_loaders
from utils.data_preprocessor import load_and_preprocess_dataset
from config import DATASET_NAME, SUBSET_NAME, EMBEDDING_DIM, BERT_MODEL_NAME, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, DEVICE

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

if __name__ == "__main__":
    train_data, test_data, user_encoder, product_encoder, preprocessed_dataset = load_and_preprocess_dataset(DATASET_NAME, SUBSET_NAME)
    
    train_loader, test_loader = get_data_loaders(train_data, test_data, BATCH_SIZE)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
    
    num_users = len(user_encoder.classes_)
    num_products = len(product_encoder.classes_)
    
    model = RecommendationModel(num_users, num_products, EMBEDDING_DIM, bert_model).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), "recommendation_model.pth")