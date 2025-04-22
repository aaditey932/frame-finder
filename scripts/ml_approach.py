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
    
    def save_model(self, model_path='./models/painting_identifier_ml.pkl'):
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
        
    def evaluate_top_n_accuracy(self, test_dataset, n=5):
        """
        Evaluate the model using Top-N accuracy.
        
        Args:
            test_dataset: Dataset of test paintings
            n: Number of top matches to consider
            
        Returns:
            Accuracy score for Top-N
        """
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
        """
        Evaluate using Mean Reciprocal Rank.
        
        Args:
            test_dataset: Dataset of test paintings
            
        Returns:
            MRR score
        """
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

def evaluate_test_folder(identifier, test_folder, dataset, k=5):
    """
    Evaluate model performance on test images from a folder.
    
    Args:
        identifier: Trained MLPaintingPredictor instance
        test_folder: Path to folder containing test images
        dataset: Original dataset for ground truth
        k: Number of top matches to consider
    """
    correct_at_1 = 0
    correct_at_k = 0
    total = 0
    reciprocal_ranks = []

    print("\n🔍 Evaluating model on test folder images...")
    
    # Check if folder exists
    if not os.path.exists(test_folder):
        print(f"⚠️ Test folder {test_folder} not found!")
        return
    
    # Loop through all test images
    for filename in tqdm(sorted(os.listdir(test_folder))):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            # Extract ground truth index from filename (e.g., "2_test_4.png" → 2)
            index = int(filename.split("_")[0])
            if index >= len(dataset):
                print(f"⚠️ Index {index} from {filename} is out of dataset range ({len(dataset)})")
                continue
                
            actual_title = dataset[index]["title"]

            # Load image
            image_path = os.path.join(test_folder, filename)
            test_image = Image.open(image_path)

            # Predict
            matches = identifier.identify_painting(test_image)
            if not matches:
                print(f"⚠️ No matches found for {filename}")
                continue
                
            top_k_titles = [m['title'] for m in matches[:k]]

            # Evaluation metrics
            if actual_title == top_k_titles[0]:
                correct_at_1 += 1
            if actual_title in top_k_titles:
                correct_at_k += 1
                rank = top_k_titles.index(actual_title) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

            total += 1

        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")

    # Results
    if total == 0:
        print("No valid test images found!")
        return None
        
    print("\n📊 Test Set Evaluation Results:")
    print(f"Total Images: {total}")
    top1_acc = correct_at_1 / total
    topk_acc = correct_at_k / total
    mrr = np.mean(reciprocal_ranks)
    
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Top-{k} Accuracy: {topk_acc:.2%}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    
    # Return metrics as a dict
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': topk_acc,
        'mrr': mrr
    }

def evaluate_ml_approach():
    """Main function to evaluate the ML approach for painting title prediction."""
    # Start timing
    start_time = time.time()
    
    # Load dataset
    print("Loading dataset...")
    try:
        ds = load_dataset("Artificio/WikiArt", split='train')
        ds = ds.select(range(100))  # Using 100 samples for evaluation
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Split dataset for training and validation
    train_size = int(0.8 * len(ds))
    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, len(ds)))
    
    print(f"Dataset split: {len(train_ds)} training samples, {len(val_ds)} validation samples")
    
    # Initialize the painting identifier
    identifier = MLPaintingPredictor(n_neighbors=5)
    
    # Check if a pre-trained model exists
    model_path = './models/MLPaintingPredictor.pkl'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            identifier.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Building new model instead...")
            identifier.build_feature_database(train_ds)
            identifier.save_model(model_path)
    else:
        print("Building features database and training model...")
        identifier.build_feature_database(train_ds)
        identifier.save_model(model_path)
    
    # Measure training time (or loading time)
    training_time = time.time() - start_time
    print(f"Model preparation time: {training_time:.2f} seconds")
    
    # Evaluate the model on the validation dataset
    print("\n=== VALIDATION SET EVALUATION ===")
    print("\nEvaluating on validation dataset...")
    val_top1_acc = identifier.evaluate_top_n_accuracy(val_ds, n=1)
    val_top5_acc = identifier.evaluate_top_n_accuracy(val_ds, n=5)
    val_mrr = identifier.evaluate_mrr(val_ds)
    
    # Evaluate on test folder (external test images)
    test_folder = "data/raw/testing_images"
    test_metrics = None
    if os.path.exists(test_folder):
        print("\n=== TEST SET EVALUATION ===")
        test_metrics = evaluate_test_folder(identifier, test_folder, ds, k=5)
    else:
        print(f"\nTest folder {test_folder} not found. Skipping test set evaluation.")
    
    # Summary of evaluation results
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Training time: {training_time:.2f}s")
    print("\nValidation Set Metrics:")
    print(f"Validation Set Size: {len(val_ds)}")
    print(f"Top-1 Accuracy: {val_top1_acc:.4f}")
    print(f"Top-5 Accuracy: {val_top5_acc:.4f}")
    print(f"Mean Reciprocal Rank: {val_mrr:.4f}")
    
    if test_metrics:
        print("\nTest Set Metrics:")
        print(f"Top-1 Accuracy: {test_metrics['top1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
        print(f"Mean Reciprocal Rank: {test_metrics['mrr']:.4f}")
    
    # Save results to CSV
    results = {
        'Approach': ['ML_Traditional'],
        'Training_Time': [training_time],
        'Val_Top1_Accuracy': [val_top1_acc],
        'Val_Top5_Accuracy': [val_top5_acc],
        'Val_MRR': [val_mrr]
    }
    
    if test_metrics:
        results.update({
            'Test_Top1_Accuracy': [test_metrics['top1_accuracy']],
            'Test_Top5_Accuracy': [test_metrics['top5_accuracy']],
            'Test_MRR': [test_metrics['mrr']]
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('ml_approach_results.csv', index=False)
    print("\nResults saved to ml_approach_results.csv")
    
    return results_df


if __name__ == "__main__":
    evaluate_ml_approach()