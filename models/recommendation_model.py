import torch
import torch.nn as nn
from transformers import BertModel


class RecommendationModel(nn.Module):
    """Neural recommender that fuses user/product embeddings with BERT text representations of product titles."""

    def __init__(self, num_users: int, num_products: int, embedding_dim: int, bert_model: BertModel):
        super().__init__()
        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.product_embeddings = nn.Embedding(num_products, embedding_dim)

        # Pre-trained language model (frozen by default; can be unfrozen for fine-tuning)
        self.bert_model = bert_model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # Fusion & prediction layers
        self.fc1 = nn.Linear(embedding_dim * 2 + bert_model.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.ReLU()

    def forward(self, user_ids: torch.LongTensor, product_ids: torch.LongTensor, product_titles_tokenised):
        """Forward pass.

        Args:
            user_ids: Tensor of shape (batch,)
            product_ids: Tensor of shape (batch,)
            product_titles_tokenised: Dict with keys input_ids, attention_mask,… (output of a HuggingFace tokenizer)
        Returns:
            Predicted rating (batch, 1)
        """
        user_emb = self.user_embeddings(user_ids)              # (batch, e)
        product_emb = self.product_embeddings(product_ids)     # (batch, e)

        # BERT CLS token embeddings for titles
        bert_outputs = self.bert_model(**product_titles_tokenised)
        text_emb = bert_outputs.last_hidden_state[:, 0, :]      # (batch, h)

        combined = torch.cat([user_emb, product_emb, text_emb], dim=1)

        x = self.activation(self.fc1(combined))
        rating = self.fc2(x)
        return rating.squeeze(-1) 