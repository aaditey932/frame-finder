import streamlit as st
from PIL import Image
import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
from scripts.dl_get_preprocessing import crop_largest_rect_from_pil
from scripts.dl_get_database import initialize_pinecone, create_pinecone_index, query_image
from scripts.dl_get_embeddings import load_clip_model, get_image_embedding
from scripts.dl_get_llm import get_art_explanation

# Prevent Streamlit from accessing torch.classes and crashing
if hasattr(torch, 'classes'):
    delattr(torch, 'classes')

# Load environment variables
load_dotenv()

# Cache OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache Pinecone initialization
@st.cache_resource
def get_pinecone_index():
    pc = initialize_pinecone(os.getenv("PINECONE_API_KEY"))
    return create_pinecone_index(pc, "frame-finder-database")

# Cache CLIP model loading
@st.cache_resource
def get_clip_model():
    return load_clip_model()

# UI setup
st.set_page_config(page_title="Frame Finder", page_icon="🎨", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🎨 Frame Finder – AI Art Identifier</h1>
    <p style='text-align: center;'>Frame-Finder is like Shazam for paintings — just snap a photo of any artwork, and the system will identify it.<br>Upload a painting and we'll tell you its story.</p>
""", unsafe_allow_html=True)

# Load resources
client = get_openai_client()
index = get_pinecone_index()
model, preprocess, device = get_clip_model()

# Upload section
uploaded_file = st.file_uploader("📤 Upload a painting", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="🖼️ Uploaded Painting", use_container_width=True)
        preprocessed_image = crop_largest_rect_from_pil(image)
        st.image(preprocessed_image, caption="🖼️ Preprocessed Painting", use_container_width=True)

    with col2:
        with st.spinner("🔍 Searching the model..."):
            embedding = get_image_embedding(preprocessed_image, model, preprocess, device)
            query_response = query_image(embedding, index, top_k=1)

        if query_response.matches:
            result = query_response.matches[0]
            st.markdown("### 🎯 Best Match")
            st.markdown(f"""
                <h3>🖌️ {result.metadata['title']}</h3>
                <p><strong>Artist:</strong> {result.metadata['artist']}<br>
                <strong>Style:</strong> {result.metadata['style']}<br>
                <strong>Genre:</strong> {result.metadata['genre']}<br>
                <strong>Similarity Score:</strong> {round(result.score, 3)}</p>
            """, unsafe_allow_html=True)

            with st.spinner("📚 Asking the art historian..."):
                explanation = get_art_explanation(result, client)
        else:
            st.error("❌ Sorry, no match was found.")
            explanation = None

    if explanation:
        st.divider()
        st.markdown(explanation, unsafe_allow_html=True)
