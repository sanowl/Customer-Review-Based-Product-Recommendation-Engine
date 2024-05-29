from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

def preprocess_data(examples):
    user_ids = examples["reviewer_id"]
    product_ids = examples["product_id"]
    ratings = examples["star_rating"]
    product_titles = examples["product_title"]
    return {"user_ids": user_ids, "product_ids": product_ids, "ratings": ratings, "product_titles": product_titles}

def load_and_preprocess_dataset(dataset_name, subset_name):
    dataset = load_dataset(dataset_name, subset_name)
    preprocessed_dataset = dataset.map(preprocess_data, remove_columns=dataset["train"].column_names)
    
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    
    preprocessed_dataset = preprocessed_dataset.map(lambda x: {
        "user_ids": user_encoder.fit_transform(x["user_ids"]),
        "product_ids": product_encoder.fit_transform(x["product_ids"]),
        "ratings": x["ratings"],
        "product_titles": x["product_titles"]
    })
    
    train_dataset = preprocessed_dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_data = train_dataset["train"]
    test_data = train_dataset["test"]
    
    return train_data, test_data, user_encoder, product_encoder, preprocessed_dataset