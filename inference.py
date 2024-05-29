import torch
from transformers import BertTokenizer

from models.recommendation_model import RecommendationModel
from utils.data_preprocessor import load_and_preprocess_dataset
from config import DATASET_NAME, SUBSET_NAME, EMBEDDING_DIM, BERT_MODEL_NAME, DEVICE

def recommend_products(user_id, top_k=5):
    user_id_enc = torch.tensor([user_encoder.transform([user_id])[0]] * num_products).to(DEVICE)
    product_ids = torch.arange(num_products).to(DEVICE)
    
    unique_product_titles = preprocessed_dataset["product_titles"].unique()
    product_titles = tokenizer(unique_product_titles, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        predicted_ratings = model(user_id_enc, product_ids, product_titles).squeeze()
        top_product_indices = torch.topk(predicted_ratings, top_k).indices
        recommended_products = unique_product_titles[top_product_indices.cpu().numpy()]
    
    return recommended_products

if __name__ == "__main__":
    _, _, user_encoder, product_encoder, preprocessed_dataset = load_and_preprocess_dataset(DATASET_NAME, SUBSET_NAME)
    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
    
    num_users = len(user_encoder.classes_)
    num_products = len(product_encoder.classes_)
    
    model = RecommendationModel(num_users, num_products, EMBEDDING_DIM, bert_model).to(DEVICE)
    model.load_state_dict(torch.load("recommendation_model.pth"))
    model.eval()
    
    user_id = "A2VNYWOPJ6U2EV"
    recommended_products = recommend_products(user_id)
    print(f"Recommended products for user {user_id}:")
    for product in recommended_products:
        print(product)