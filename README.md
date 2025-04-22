# 🖼️ Frame-Finder

## An AI-Powered Painting Recognition and Explanation System

**Frame-Finder**  is like Shazam for paintings — just snap a photo of any artwork, and the system will identify it or find visually and stylistically similar pieces from a curated database. Powered by CLIP and vector search, Frame-Finder also provides insightful art explanations using OpenAI GPT-4o-mini. It’s built for anyone curious about the art around them — in museums, galleries, or on the street.

---

## 🖼️ Dataset

The project uses a subset of the **[WikiArt Dataset]([https://www.wikiart.org/](https://huggingface.co/datasets/Artificio/WikiArt))**.  
50 painting images and associated metadata were downloaded and processed using:

📄 `dl_get_data.py`

It contains:

- Painting titles  
- Artist names  
- Styles (e.g., Impressionism, Baroque)  
- Genres (e.g., Portrait, Landscape)  
- Images

Images are saved in a local directory and metadata is stored as a structured DataFrame.

---

## 🧪 Approaches Implemented

### [`scripts/`](scripts)

All scripts are modular and include utility functions for preprocessing, embedding generation, and evaluation.
> **Note:** Only the **Naïve** and **Similarity-Based ML** models generate `.pkl` checkpoint files for reuse. The deep learning approach doesn't involve training a custom neural network so no model weights are saved.
---

### ✅ 1. **Naïve Approach**

This approach doesn’t rely on deep learning or semantic embeddings. Instead, it uses **color histogram features** — a classic computer vision technique — to compare visual similarity across paintings.

#### 🔍 How it works:
- Extracts **color histograms** (RGB) from the uploaded painting using OpenCV
- Uses **histogram intersection** to calculate similarity scores
- Ranks paintings by visual similarity and returns the top N matches

#### 📌 When it’s useful:
- Works well for matching images with similar colors and styles
- Fast to compute and easy to interpret

#### ⚠️ Limitations:
- Does not capture deeper semantic or structural patterns in the artwork
- Sensitive to lighting changes and cropping in user photos

📄 Script: `naive_approach.py`  
📦 Model saved as: `models/NaivePaintingPredictor.pkl`

--- 

### ✅ 2. **Traditional ML Approach**

This approach uses a **rich combination of features** from the image — including color histograms, textures, edge patterns, and layout — to find visually similar paintings using a classic ML pipeline.

#### 🧠 What makes it “ML”:
Unlike the naïve model that only compares simple color histograms, this version:
- Extracts **multiple feature types** (color, texture, edge, layout)
- Standardizes and reduces dimensionality using **PCA**
- Uses **cosine similarity with a Nearest Neighbors model** to rank visually similar paintings

#### 🧩 Features Extracted:
| Feature Type     | Description                                                    |
|------------------|----------------------------------------------------------------|
| Color Histogram  | HSV color distribution across the entire image                 |
| Texture Features | Haralick texture descriptors using GLCM (contrast, energy, etc.) |
| Edge Features    | Edge direction histogram and edge pixel density               |
| Color Layout     | Average HSV values from a 4x4 grid layout                      |

#### 🔍 How it works:
1. For each image, extract the 4 types of features
2. Standardize and reduce dimensions using PCA (50 components max)
3. Fit a Nearest Neighbors model using cosine similarity
4. For any query image, return the top N most similar paintings

#### 📌 When it’s useful:
- Captures detailed visual structure of paintings
- Works even when image is cropped or partially obscured
- Performs well on visually distinctive artworks

#### ⚠️ Limitations:
- Still limited to low-level visual features (no semantic understanding)
- Sensitive to noise, lighting, and image quality

📄 Script: `ml_approach.py`  
📦 Model saved as: `models/MLPaintingPredictor.pkl`

---

### ✅ 3. **Deep Learning Approach**

This is the most advanced and scalable approach in Frame-Finder. It uses **OpenAI’s CLIP API** to understand images, **Pinecone** as a fast vector database for similarity search, and **GPT-based LLMs** to explain the artwork.

📄 Core Scripts:
- `dl_get_embeddings.py`: Loads a CLIP model (ResNet 101), generates embeddings
- `dl_get_database.py`: Manages Pinecone setup and querying
- `dl_get_preprocessing.py`: Crops and preprocesses input images for consistency
- `dl_get_data.py`: Downloads and organizes reference dataset
- `dl_get_llm`: Generates an art explanation using a Large Language Model (OpenAI GPT 4o-mini).
- `app.py` (Streamlit UI)

#### 🧠 What makes it “deep learning”:
- Uses **CLIP**, a pre-trained model that jointly understands images and text
- Transforms the input painting into a **semantic embedding**
- Searches a **vector index of known artworks** for the closest matches
- Uses GPT to generate a **natural-language explanation** of the best match

#### 🔍 How it works:

1. **Preprocessing**: The image is cropped to its largest visible region (`crop_largest_rect_from_pil`).
2. **Embedding**: The processed image is passed through the CLIP model to get a semantic vector (`get_image_embedding`).
3. **Search**: The vector is matched against all stored embeddings in **Pinecone** (`query_image`) to retrieve top 1 similar paintings.
4. **Explain**: Once a match is found, its metadata is passed to a GPT-based LLM (`get_art_explanation`) to generate a rich interpretation.

#### ⚙️ Technologies Used

| Component         | Description                                      |
|------------------|--------------------------------------------------|
| **CLIP ResNet 101**          | Pre-trained vision-language model for embeddings |
| **Pinecone**      | Vector database for semantic search |
| **Streamlit**     | UI for image upload and visualization            |
| **OpenAI GPT**    | Generates natural-language insights and context  |

#### 📌 Why it’s powerful:
- **Semantic** understanding of art (not just pixels or colors)
- Scalable to thousands of paintings with near real-time search
- **Explains** art in natural language, enhancing user experience

#### ⚠️ Limitations:
- Performance limited by the quality of reference embeddings and uploaded images
- CLIP is fixed (not fine-tuned on art-specific data)

---

## 📊 Evaluation

To assess the performance of each approach, we used a test set of user-uploaded painting images and measured how accurately the system could identify the correct painting (or a close match) from the dataset.

### 🎯 Metrics Used

| Metric                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Hit@K**               | Whether the correct painting is in the top K results                        |
| **Recall@K**            | Proportion of correct items retrieved within top K                          |
| **Mean Average Precision (mAP)** | Average precision across positions where correct items are found       |
| **Top-1 Similarity Score** | Similarity score of the best-ranked painting (0 to 1, higher is better) |

### 🧪 Results Overview

| Approach                    | Hit@5 (%) | Recall@5 | mAP    | Average Top-1 Similarity |
|-----------------------------|-----------|----------|--------|------------------------|
| Naïve Color Histogram       |   `0.00%`   |  `0.0000`   | `0.0000` |        `12.0511`          |
| ML – Traditional Features   |   `30.00%`   |  `0.3000`   | `0.1617` |        `0.4810`          |
| Deep Learning (CLIP + DB)   |   `95.00%`   |  `0.9500`   | `0.9500` |        `0.8900`          |

### 🧠 Insights

- **Naïve Approach** tends to perform poorly on complex or cropped images, but can still be useful for very distinctive artworks.
- **ML Approach** offers balanced performance and is more robust to partial inputs due to the variety of features used (color, texture, edge, layout).
- **CLIP + Pinecone** consistently performs best in terms of both accuracy and explanation, making it ideal for real-time art recognition.
- 
---

## 🚀 Try the Live App

Want to see Frame-Finder in action?  
Upload any painting and get instant recognition and an AI-generated explanation.

👉 **[Launch the App](https://frame-finder-ai.streamlit.app)**

---

## 🎨 Sample Use Case

1. User uploads a photo of a painting (e.g., taken at a museum)
2. Frame-Finder preprocesses and embeds the image
3. Top 1 visually similar paintings from the WikiArt dataset are retrieved
4. System highlights the best match and generates an explanation

---

## ⚖️ Ethical Considerations

When building and deploying an AI-powered art recognition system, several ethical issues must be addressed:

### 1. **Copyright and Intellectual Property**
- The system must respect artists' intellectual property rights and not facilitate copyright infringement
- Attribution should always be given to artists when their work is identified
- The system should not enable unauthorized reproduction or commercial use of artworks

### 2. **Data Privacy**
- User-uploaded images should not be retained or used for purposes beyond identification
- Clear privacy policies should explain how user data is handled
- Consider implementing on-device processing options to minimize data collection

### 3. **Bias in Art Recognition**
- CLIP and other AI models may have biases toward Western/canonical art
- The system should clearly communicate its limitations in recognizing non-Western or contemporary art
- Regularly audit the system for biases in recognition accuracy across different art styles

### 4. **Impact on Art Education and Appreciation**
- The system should enhance, not replace, human expertise in art interpretation
- Explanations should encourage deeper engagement with art rather than just quick identification
- Consider partnering with art educators to ensure the system supports learning objectives

---

## 🌐 Future Improvements

- Add user feedback loop for learning preferences  
- Scale to full WikiArt dataset (~80K paintings)  
- Add OCR + context detection for mixed media art  
- Fine-tune CLIP on art-specific embeddings  

---

## 🖥️ Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the system
```bash
streamlit run app.py
```

> Ensure your Pinecone API key and OpenAI API key are securely added as environment variables.

