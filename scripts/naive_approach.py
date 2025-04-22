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
    

def evaluate_naive_approach(test_folder, predictor, metadata_path, results_csv, k=5):
    """Evaluate the naive approach for painting title prediction."""
    recall_scores = []
    hit_scores = []
    avg_precision_scores = []
    similarity_scores = []
    rows = []
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    for filename in tqdm(sorted(os.listdir(test_folder))):
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            continue
        
        try:
            index_gt = int(filename.split("_")[0])  # e.g., 2_test_3.png -> 2
            img_path = os.path.join(test_folder, filename)
            image = Image.open(img_path)
            
            # Get matches using naive approach
            matches = predictor.identify_painting(image)
            retrieved_titles = [m['title'] for m in matches][:k]  # Limit to top k
            top1_title = retrieved_titles[0] if retrieved_titles else ""
            top1_score = matches[0]['score'] if matches else 0.0
            similarity_scores.append(top1_score)
            
            # Ground truth title from metadata.csv
            ground_truth_title = metadata_df.loc[index_gt, 'title']
            
            # Evaluation Metrics
            hit = int(ground_truth_title in retrieved_titles)
            recall = 1.0 if ground_truth_title in retrieved_titles else 0.0
            
            # Average Precision (AP)
            ap = 0.0
            correct = 0
            for i, title in enumerate(retrieved_titles):
                if title == ground_truth_title:
                    correct += 1
                    ap += correct / (i + 1)
            ap = ap / correct if correct else 0.0
            
            # Save metrics
            hit_scores.append(hit)
            recall_scores.append(recall)
            avg_precision_scores.append(ap)
            
            # Save row
            rows.append({
                "filename": filename,
                "ground_truth_title": ground_truth_title,
                "top1_title": top1_title,
                "hit@k": hit,
                "recall@k": recall,
                "mAP": ap,
                "top1_similarity_score": top1_score
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Save results to CSV
    df = pd.DataFrame(rows)
    df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to {results_csv}")
    
    # Final summary
    print("\n📊 Evaluation Summary")
    print(f"Total Images Evaluated: {len(rows)}")
    print(f"Top-{k} Accuracy (Hit Rate): {np.mean(hit_scores):.2%}")
    print(f"Recall@{k}: {np.mean(recall_scores):.4f}")
    print(f"Mean Average Precision (mAP): {np.mean(avg_precision_scores):.4f}")
    print(f"Average Similarity Score (Top-1): {np.mean(similarity_scores):.4f}")

def main():
    """Main function to run the script."""
    TEST_FOLDER = "data/raw/testing_images"
    METADATA_PATH = "data/raw/metadata.csv"
    RESULTS_CSV = "data/output/naive_approach_results.csv"
    TOP_K = 5
    
    # Load dataset for building the database
    print("Loading dataset...")
    try:
        ds = load_dataset("Artificio/WikiArt", split='train')
        ds = ds.select(range(100))  # Using 100 samples for evaluation
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Initialize the naive painting predictor
    predictor = NaivePaintingPredictor(n_neighbors=TOP_K)
    
    # Build database
    print("Building database with naive approach...")
    predictor.build_database(ds)
    
    # Save model
    predictor.save_model()
    
    # Evaluate
    evaluate_naive_approach(TEST_FOLDER, predictor, METADATA_PATH, RESULTS_CSV, k=TOP_K)
    
if __name__ == "__main__":
    main()