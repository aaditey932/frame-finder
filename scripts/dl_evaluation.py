import os
import numpy as np
import pandas as pd
import csv
from PIL import Image
from tqdm import tqdm

from dl_get_preprocessing import crop_largest_rect_from_pil
from dl_get_database import initialize_pinecone, create_pinecone_index, query_image
from dl_get_embeddings import load_clip_model, get_image_embedding

# ------------------- Evaluation Script -------------------
def evaluate_retrieval(test_folder, results_csv, metadata_df, model, preprocess, device, index, k=5):
    """Evaluate the retrieval performance of the deep learning approach.
    
    Args:
        test_folder (str): Path to folder containing test images
        results_csv (str): Path to save evaluation results
        metadata_df (pd.DataFrame): DataFrame containing ground truth metadata
        model: CLIP model for generating embeddings
        preprocess: Preprocessing function for CLIP
        device: Device to run model on (CPU/GPU)
        index: Pinecone index for similarity search
        k (int): Top-k results to consider for evaluation
    
    Returns:
        None (saves results to CSV and prints summary)
    """
    recall_scores = []
    hit_scores = []
    avg_precision_scores = []
    similarity_scores = []

    rows = []

    for filename in tqdm(sorted(os.listdir(test_folder))):
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            index_gt = int(filename.split("_")[0])  # e.g., 2_test_3.png -> 2
            img_path = os.path.join(test_folder, filename)
            image = Image.open(img_path)
            image = crop_largest_rect_from_pil(image)

            # Get embedding
            embedding = get_image_embedding(image, model, preprocess, device)

            # Query Pinecone
            results = query_image(embedding, index, top_k=k)
            matches = results['matches']
            retrieved_titles = [m['metadata']['title'] for m in matches]
            top1_title = retrieved_titles[0]
            top1_score = matches[0]['score']
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
    RESULTS_CSV = "data/output/dl_approach_results.csv"
    INDEX_NAME = "frame-finder-database"
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    TOP_K = 5

    # Load metadata
    metadata_df = pd.read_csv(METADATA_PATH)

    # Load model and Pinecone
    model, preprocess, device = load_clip_model()
    pc = initialize_pinecone(PINECONE_API_KEY)
    index = create_pinecone_index(pc, INDEX_NAME)
    evaluate_retrieval(TEST_FOLDER, RESULTS_CSV, metadata_df, model, preprocess, device, index, k=TOP_K)
    
if __name__ == "__main__":
    main()
