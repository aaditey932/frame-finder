import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Optional


def create_directories(base_dir: str) -> str:
    """
    Create necessary directories for storing images and metadata.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Path to the images directory
    """
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def download_wikiart_dataset(
    output_dir: str, 
    num_samples: Optional[int] = None, 
    split: str = 'train'
) -> pd.DataFrame:
    """
    Download WikiArt dataset, save images, and create metadata CSV.
    
    Args:
        output_dir: Directory to save images and metadata
        num_samples: Number of samples to download (None for all)
        split: Dataset split to use
        
    Returns:
        DataFrame containing metadata
    """
    # Create directories
    images_dir = create_directories(output_dir)
    
    # Load dataset
    ds = load_dataset("Artificio/WikiArt", split=split)
    if num_samples is not None:
        ds = ds.select(range(num_samples))
    print(f"Loaded dataset: {ds}")
    
    # Save images and collect metadata
    metadata = []
    for i, example in enumerate(tqdm(ds, desc="Downloading images")):
        img: Image.Image = example["image"]
        img_filename = f"{i}_{example['title'].replace('/', '_')}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        
        # Save image
        img.save(img_path)
        
        # Collect metadata
        metadata.append({
            "file": img_filename,
            "artist": example["artist"],
            "style": example["style"],
            "genre": example["genre"],
            "title": example["title"],
        })
    
    # Create DataFrame and save as CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")
    
    return metadata_df


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load metadata from CSV file.
    
    Args:
        metadata_path: Path to metadata CSV file
        
    Returns:
        DataFrame containing metadata
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    return pd.read_csv(metadata_path)


def main():
    """Main function to run the script."""
    # Configure parameters
    output_dir = "./data/wikiart_export"
    num_samples = 10  # Set to None to download the entire dataset
    
    # Download dataset and save
    metadata = download_wikiart_dataset(
        output_dir=output_dir,
        num_samples=num_samples
    )
    print(f"Downloaded {len(metadata)} samples")


if __name__ == "__main__":
    main()