import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import pickle

class NaivePaintingPredictor:
    def __init__(self, n_neighbors=5):
        """Initialize the Naive Painting Title Predictor."""
        self.n_neighbors = n_neighbors
        self.features_df = None
        
    def extract_color_histogram(self, image, bins=32):
        """Extract simple color histogram features from an image."""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize image for consistency
        image = cv2.resize(image, (224, 224))
        
        # Calculate histograms for each channel
        b_hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        g_hist = cv2.calcHist([image], [1], None, [bins], [0, 256])
        r_hist = cv2.calcHist([image], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        cv2.normalize(b_hist, b_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Concatenate histograms
        hist_features = np.concatenate([b_hist, g_hist, r_hist]).flatten()
        return hist_features
    
    def build_database(self, dataset):
        """Build a database of features from all paintings in the dataset."""
        features_list = []
        
        print("Extracting color histograms from paintings...")
        for i, example in enumerate(tqdm(dataset)):
            try:
                # Get image and metadata
                image = example["image"]
                
                # Extract features
                features = self.extract_color_histogram(image)
                
                # Store features with metadata
                features_dict = {
                    'title': example['title'],
                    'artist': example['artist'],
                    'style': example['style'],
                    'genre': example['genre'],
                    'date': example.get('date', 'Unknown'),
                    'filename': example.get('filename', f'image_{i}'),
                    'features': features
                }
                
                features_list.append(features_dict)
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(features_list)
        return self.features_df

    def identify_painting(self, image):
        """Identify a painting from an image using color histogram distance."""
        if self.features_df is None or len(self.features_df) == 0:
            print("No database built yet!")
            return []
            
        try:
            # Extract features
            query_features = self.extract_color_histogram(image)
            
            # Calculate distances for all features in the database
            distances = []
            for idx, row in self.features_df.iterrows():
                db_features = row['features']
                # Calculate histogram intersection (higher is better)
                intersection = cv2.compareHist(
                    query_features.reshape((len(query_features), 1)), 
                    db_features.reshape((len(db_features), 1)), 
                    cv2.HISTCMP_INTERSECT
                )
                # Convert to distance (lower is better)
                distance = 1.0 - intersection
                distances.append((distance, idx))
            
            # Sort by distance (lowest first)
            distances.sort()
            
            # Get top N matches
            matches = []
            for i, (dist, idx) in enumerate(distances[:self.n_neighbors]):
                match_info = self.features_df.iloc[idx].copy()
                # Convert distance to similarity score (0-1, higher is better)
                similarity = 1.0 - dist
                match_info['score'] = similarity
                match_info['rank'] = i + 1
                # Drop features to avoid large dictionaries
                match_info = match_info.drop(['features'])
                matches.append(match_info)
            
            return matches
            
        except Exception as e:
            print(f"Error in identification: {e}")
            return []
        
    def save_model(self, model_path='./models/NaivePaintingPredictor.pkl'):
        """Save the feature database to a pickle file."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model_data = {
            'features_df': self.features_df
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Naive model saved to {model_path}")

    def load_model(self, model_path='./models/naive_painting_model.pkl'):
        """Load the feature database from a pickle file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.features_df = model_data['features_df']
        print(f"Naive model loaded from {model_path}")
    
    def evaluate_top_n_accuracy(self, test_dataset, n=5):
        """Evaluate the model using Top-N accuracy."""
        correct = 0
        total = 0
        
        print(f"Evaluating Top-{n} accuracy...")
        for example in tqdm(test_dataset):
            try:
                # Get actual title
                actual_title = example['title']
                
                # Get predictions
                matches = self.identify_painting(example['image'])
                
                # Check if actual title is in top N predictions
                predicted_titles = [match['title'] for match in matches[:n]]
                if actual_title in predicted_titles:
                    correct += 1
                
                total += 1
            except Exception as e:
                print(f"Error evaluating example: {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"Top-{n} accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    def evaluate_mrr(self, test_dataset):
        """Evaluate using Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        print("Evaluating Mean Reciprocal Rank...")
        for example in tqdm(test_dataset):
            try:
                # Get actual title
                actual_title = example['title']
                
                # Get predictions
                matches = self.identify_painting(example['image'])
                predicted_titles = [match['title'] for match in matches]
                
                # Find rank of correct title
                if actual_title in predicted_titles:
                    rank = predicted_titles.index(actual_title) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
                    
            except Exception as e:
                print(f"Error evaluating example: {e}")
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks)
        print(f"Mean Reciprocal Rank: {mrr:.4f}")
        return mrr

def evaluate_naive_approach():
    """Evaluate the naive approach for painting title prediction."""
    # Load dataset
    print("Loading dataset...")
    try:
        ds = load_dataset("Artificio/WikiArt", split='train')
        ds = ds.select(range(100))  # Using 100 samples for evaluation
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Split dataset for training and testing
    train_size = int(0.8 * len(ds))
    train_ds = ds.select(range(train_size))
    test_ds = ds.select(range(train_size, len(ds)))
    
    print(f"Dataset split: {len(train_ds)} training samples, {len(test_ds)} testing samples")
    
    # Initialize the naive painting predictor
    predictor = NaivePaintingPredictor(n_neighbors=5)
    
    # Build database
    print("Building database with naive approach...")
    predictor.build_database(train_ds)

    # Save model
    predictor.save_model()
    
    # Evaluate
    print("\n=== NAIVE APPROACH EVALUATION ===")
    top1_acc = predictor.evaluate_top_n_accuracy(test_ds, n=1)
    top5_acc = predictor.evaluate_top_n_accuracy(test_ds, n=5)
    mrr = predictor.evaluate_mrr(test_ds)
    
    # Summary
    print("\n=== EVALUATION SUMMARY (NAIVE APPROACH) ===")
    print(f"Test Dataset Size: {len(test_ds)}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")


if __name__ == "__main__":
    evaluate_naive_approach()