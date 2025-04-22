import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import pickle
import time
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

class MLPaintingPredictor:
    def __init__(self, n_neighbors=5):
        """
        Initialize the ML Painting Predictor using traditional ML.
        
        Args:
            n_neighbors: Number of neighbors for similarity search
        """
        self.n_neighbors = n_neighbors
        self.scaler = None
        self.pca_model = None
        self.nearest_neighbors = None
        self.features_df = None
        
    def extract_color_histogram(self, image, bins=32):
        """Extract color histogram features from an image."""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to HSV color space (better for color-based features)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Concatenate histograms
        hist_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        return hist_features
    
    def extract_texture_features(self, image):
        """Extract texture features using Haralick texture features."""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        glcm = cv2.GaussianBlur(gray_image, (7, 7), 0)
        
        # Scale to 32 levels for GLCM
        levels = 32
        glcm = (gray_image.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)
        
        # Compute GLCM
        distances = [1, 3]
        angles = [0, np.pi/4]
        glcm_matrix = graycomatrix(glcm, distances=distances, angles=angles, 
                            levels=levels, symmetric=True, normed=True)
        
        # Extract properties
        contrast = graycoprops(glcm_matrix, 'contrast').flatten()
        dissimilarity = graycoprops(glcm_matrix, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm_matrix, 'homogeneity').flatten()
        energy = graycoprops(glcm_matrix, 'energy').flatten()
        correlation = graycoprops(glcm_matrix, 'correlation').flatten()
        
        # Concatenate features
        texture_features = np.hstack([
            contrast, dissimilarity, homogeneity, energy, correlation
        ])
        
        return texture_features
    
    def extract_edge_features(self, image):
        """Extract edge-based features using edge detection."""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray_image, 100, 200)
        
        # Calculate histogram of edge directions
        gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Only consider angles where there's an edge
        edge_mask = edges > 0
        edge_angles = angle[edge_mask]
        
        # Create histogram of edge directions
        if len(edge_angles) > 0:
            hist, _ = np.histogram(edge_angles, bins=18, range=(0, 360))
            hist = hist / max(np.sum(hist), 1)  # Normalize
        else:
            hist = np.zeros(18)
            
        # Calculate edge density (percentage of edge pixels)
        edge_density = np.sum(edge_mask) / (edges.shape[0] * edges.shape[1])
        
        # Return edge features
        return np.append(hist, edge_density)
    
    def extract_color_layout(self, image, grid_size=4):
        """Extract color layout features by dividing the image into a grid."""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Resize image to ensure consistent size
        image = cv2.resize(image, (224, 224))
        
        # Convert to HSV for better color representation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Initialize features
        features = []
        
        # Divide image into grid
        h, w = image.shape[:2]
        h_step = h // grid_size
        w_step = w // grid_size
        
        # Extract average color from each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                # Define cell region
                cell = hsv_image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                
                # Calculate average HSV values
                avg_h = np.mean(cell[:, :, 0])
                avg_s = np.mean(cell[:, :, 1])
                avg_v = np.mean(cell[:, :, 2])
                
                # Add to features
                features.extend([avg_h, avg_s, avg_v])
        
        return np.array(features)
    
    def extract_features(self, image):
        """Extract combined features from an image."""
        # Resize image for consistency if it's a PIL image
        if isinstance(image, Image.Image):
            image = image.resize((224, 224))
        else:
            image = cv2.resize(image, (224, 224))
        
        # Extract different feature types
        color_features = self.extract_color_histogram(image)
        texture_features = self.extract_texture_features(image)
        edge_features = self.extract_edge_features(image)
        layout_features = self.extract_color_layout(image)
        
        # Combine all features
        combined_features = np.concatenate([
            color_features,     # Color distribution
            texture_features,   # Texture patterns
            edge_features,      # Edge information
            layout_features     # Spatial color organization
        ])
        
        return combined_features
    
    def build_feature_database(self, dataset, val_size=0.2, random_state=42):
        """Build a database of features from all paintings in the dataset."""
        features_list = []
        
        print("Extracting features from paintings...")
        for i, example in enumerate(tqdm(dataset)):
            try:
                # Get image and metadata
                image = example["image"]
                
                # Extract features
                features = self.extract_features(image)
                
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
        
        if len(features_list) == 0:
            raise ValueError("No features could be extracted from the dataset")
            
        # Extract features array
        X = np.stack(self.features_df['features'].values)
        
        # Standardize features
        print("Standardizing features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        print("Applying PCA...")
        pca_components = min(50, X_scaled.shape[1], X_scaled.shape[0] - 1)
        self.pca_model = PCA(n_components=pca_components)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Print explained variance
        explained_var = np.sum(self.pca_model.explained_variance_ratio_)
        print(f"PCA with {pca_components} components explains {explained_var:.2%} of variance")
        
        # Build nearest neighbors model
        print("Building nearest neighbors model...")
        self.nearest_neighbors = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(features_list)), metric='cosine')
        self.nearest_neighbors.fit(X_pca)
        
        # Store PCA-transformed features
        self.features_df['features_pca'] = list(X_pca)
        
        return self.features_df
    
    def identify_painting(self, image):
        """Identify a painting from an image."""
        try:
            # Extract features
            features = self.extract_features(image)
            
            # Scale and apply PCA
            features_scaled = self.scaler.transform([features])
            features_pca = self.pca_model.transform(features_scaled)
            
            # Find nearest neighbors
            distances, indices = self.nearest_neighbors.kneighbors(features_pca)
            
            # Get matches
            matches = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                # Convert distance to similarity score (0-1)
                similarity = 1 - dist
                
                match_info = self.features_df.iloc[idx].copy()
                match_info['score'] = similarity
                match_info['rank'] = i + 1
                
                # Drop features to avoid large dictionaries
                match_info = match_info.drop(['features', 'features_pca'])
                
                matches.append(match_info)
            
            return matches
        except Exception as e:
            print(f"Error in identification: {e}")
            return []
    
    def save_model(self, model_path='./models/MLPaintingPredictor.pkl'):
        """Save the ML model to a file."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'scaler': self.scaler,
            'pca_model': self.pca_model,
            'nearest_neighbors': self.nearest_neighbors,
            'features_df': self.features_df
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='./models/painting_identifier_ml.pkl'):
        """Load the ML model from a file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.pca_model = model_data['pca_model']
        self.nearest_neighbors = model_data['nearest_neighbors']
        self.features_df = model_data['features_df']
        
        print(f"Model loaded from {model_path}")
        
def evaluate_ml_approach(test_folder, predictor, metadata_path, results_csv, k=5):
    """Evaluate the ML approach for painting title prediction."""
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
            
            # Get matches using ML approach
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
    RESULTS_CSV = "data/output/ml_approach_results.csv"
    TOP_K = 5
    
    # Load dataset for building the database
    print("Loading dataset...")
    try:
        ds = load_dataset("Artificio/WikiArt", split='train')
        ds = ds.select(range(100))  # Using 100 samples for evaluation
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Initialize the ML painting predictor
    predictor = MLPaintingPredictor(n_neighbors=TOP_K)
    
    # Build database
    print("Building database with ML approach...")
    predictor.build_feature_database(ds)
    
    # Save model
    predictor.save_model()
    
    # Evaluate
    evaluate_ml_approach(TEST_FOLDER, predictor, METADATA_PATH, RESULTS_CSV, k=TOP_K)

if __name__ == "__main__":
    main()