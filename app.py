import streamlit as st
from PIL import Image
import os
from openai import OpenAI
from dotenv import load_dotenv
import sys
import torch

from scripts.dl_get_preprocessing import crop_largest_rect_from_pil
from scripts.dl_get_database import initialize_pinecone, create_pinecone_index, query_image
from scripts.dl_get_embeddings import load_clip_model, get_image_embedding
from scripts.dl_get_llm import get_art_explanation



# Prevent Streamlit from accessing torch.classes and crashing
if hasattr(torch, 'classes'):
    delattr(torch, 'classes')

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "frame-finder-database"

st.set_page_config(page_title="Frame Finder", page_icon="🎨", layout="wide")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = initialize_pinecone(PINECONE_API_KEY)
index = create_pinecone_index(pc, INDEX_NAME)

model, preprocess, device = load_clip_model()

# -------------------- STREAMLIT UI --------------------

st.markdown("""
    <style>
    .title { font-size: 38px; font-weight: bold; }
    .subtitle { font-size: 20px; margin-top: -10px; color: #777; }
    .card {
        background-color: #fafafa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎨 Frame Finder – AI Art Identifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Frame-Finder is like Shazam for paintings — just snap a photo of any artwork, and the system will identify it. \n Upload a painting and we'll tell you its story</div>", unsafe_allow_html=True)
st.markdown("")

uploaded_file = st.file_uploader("📤 Upload a painting", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])  # Wider column for best match

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
            <div class="card">
                <h3>🖌️ <b>{result.metadata['title']}</b></h3>
                <p><b>Artist:</b> {result.metadata['artist']}<br>
                <b>Style:</b> {result.metadata['style']}<br>
                <b>Genre:</b> {result.metadata['genre']}<br>
                <b>Similarity Score:</b> {round(result.score, 3)}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("📚 Asking the art historian..."):
                explanation = get_art_explanation(result, client)
        else:
            st.error("❌ Sorry, no match was found.")
            explanation = None

        if explanation:
            st.divider()
            st.markdown(explanation, unsafe_allow_html=True)