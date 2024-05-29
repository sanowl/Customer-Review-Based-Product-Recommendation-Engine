import torch
import torch.nn as nn
from transformers import BertModel

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim, bert_model):
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