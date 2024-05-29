# Customer Review-Based Product Recommendation Engine

This project implements an advanced recommendation engine using the "amazon_us_reviews" dataset from the Hugging Face datasets library. The recommendation engine utilizes deep learning techniques, including embeddings and the BERT model, to provide personalized product recommendations based on customer reviews.

## Features

- Utilizes the "amazon_us_reviews" dataset from Hugging Face, specifically the "Video_Games_v1_00" subset.
- Preprocesses the dataset to extract user IDs, product IDs, ratings, and product titles.
- Implements a custom neural network architecture that combines user embeddings, product embeddings, and BERT-based product title features.
- Trains the recommendation model using the preprocessed dataset and optimizes the model parameters.
- Evaluates the trained model on a test set and calculates the mean squared error loss.
- Provides a function to generate top-k product recommendations for a given user.

## Prerequisites

- Python 3.x
- PyTorch
- Hugging Face Datasets
- Hugging Face Transformers

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/sanowl/Customer-Review-Based-Product-Recommendation-Engine.git
   ```

2. Install the required libraries:

   ```
   pip install torch datasets transformers
   ```

## Usage

1. Run the script:

   ```
   python recommendation_engine.py
   ```

   The script will load the dataset, preprocess it, train the recommendation model, evaluate the model on the test set, and provide an example usage of generating recommendations for a specific user.

2. Customize the dataset and model:
   - You can modify the code to use a different subset of the "amazon_us_reviews" dataset or a different dataset altogether. Update the `load_dataset` function call with the appropriate dataset name and subset.
   - Adjust the model architecture, hyperparameters, and training settings in the code to suit your specific requirements.

3. Integrate the recommendation engine into your application:
   - Use the trained model to generate product recommendations for users in your application.
   - Modify the `recommend_products` function to accept user input and return the recommended products.

## Dataset

The "amazon_us_reviews" dataset from Hugging Face is used in this project. It contains customer reviews from Amazon across various product categories. The "Video_Games_v1_00" subset is specifically used in this example, but you can choose a different subset or dataset based on your requirements.

## Model Architecture

The recommendation model combines user embeddings, product embeddings, and BERT-based product title features to predict ratings. The architecture consists of the following components:
- User Embedding Layer: Maps user IDs to dense vector representations.
- Product Embedding Layer: Maps product IDs to dense vector representations.
- BERT Model: Extracts features from product titles using the pre-trained BERT model.
- Fully Connected Layers: Combines user embeddings, product embeddings, and BERT features to predict ratings.

## Evaluation

The trained model is evaluated on a test set using the mean squared error (MSE) loss. The test loss is printed after training.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The "amazon_us_reviews" dataset is provided by Hugging Face.
- The BERT model is developed by Google and made available through the Hugging Face Transformers library.

## Contact

For any questions or inquiries, please contact sanowl98@gmail.com

---
