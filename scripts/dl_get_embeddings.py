import os
import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Union, Optional, Tuple


def load_clip_model(model_name: str = 'RN101', device: Optional[str] = None) -> Tuple:
    """
    Load a CLIP model.
    
    Args:
        model_name: Name of the CLIP model to load
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, preprocess, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model: {model_name} on {device}")
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def get_image_embedding(
    image: Image, 
    model: torch.nn.Module, 
    preprocess, 
    device: str
) -> List[float]:
    """
    Get CLIP embedding for an image.
    
    Args:
        image_path: Path to the image file
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Image embedding as a list of floats
    """
    # Load and preprocess image
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    # Generate embedding
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
        # Convert to numpy array, flatten, and then to list
        embedding_list = image_embedding.cpu().numpy().flatten().tolist()
    
    return embedding_list


def generate_embeddings_for_dataset(
    metadata_df: pd.DataFrame,
    images_dir: str,
    model_name: str = 'RN101',
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate embeddings for all images in the dataset.
    
    Args:
        metadata_df: DataFrame with metadata including filenames
        images_dir: Directory containing images
        model_name: CLIP model name to use
        device: Device to run inference on
        
    Returns:
        DataFrame with original metadata plus embeddings
    """
    # Load CLIP model
    model, preprocess, device = load_clip_model(model_name, device)
    
    # Create a copy of the DataFrame to add embeddings
    result_df = metadata_df.copy()
    
    # Store embeddings
    embeddings = []
    
    # Process each image
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Generating embeddings"):
        # Construct image path
        image_path = os.path.join(images_dir, row["file"])
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            embeddings.append(None)
            continue
        
        # Get embedding
        image = Image.open(image_path)
        embedding = get_image_embedding(image, model, preprocess, device)
        embeddings.append(embedding)
    
    # Add embeddings to DataFrame
    result_df["image_embedding"] = embeddings
    
    # Filter out rows with missing embeddings
    valid_df = result_df.dropna(subset=["image_embedding"])
    if len(valid_df) < len(result_df):
        print(f"Warning: {len(result_df) - len(valid_df)} images could not be processed")
    
    return valid_df

def main():
    """Main function to run the script."""
    # Configure parameters
    metadata_path = "./data/wikiart_export/metadata.csv"
    images_dir = "./data/wikiart_export/images"
    clip_model_name = 'RN101'
    
    # Load metadata
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"Loaded metadata for {len(metadata_df)} images")
    except FileNotFoundError:
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    # Generate embeddings
    df_with_embeddings = generate_embeddings_for_dataset(
        metadata_df=metadata_df,
        images_dir=images_dir,
        model_name=clip_model_name
    )
    
    # Display sample
    if len(df_with_embeddings) > 0:
        print("\n🔢 Sample Embedding Vector:")
        sample_embedding = df_with_embeddings["image_embedding"].iloc[0]
        print(sample_embedding[:10])  # Show first 10 dimensions
        
        # Save DataFrame with embeddings
        output_path = "./data/wikiart_export/embeddings_data.csv"
        df_with_embeddings.to_csv(output_path, index=False)
        print(f"Saved {len(df_with_embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    main()