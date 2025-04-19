import streamlit as st
from PIL import Image
import torch
import clip
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "frame-finder-database"

st.set_page_config(page_title="Frame Finder", page_icon="🎨", layout="wide")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@st.cache_resource
def load_clip():
    model, preprocess = clip.load("RN101", device="cpu")
    return model, preprocess

model, preprocess = load_clip()

def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        return model.encode_image(image).squeeze().cpu().numpy()

def get_art_explanation(results):
    prompt = f"""
You are an expert art historian and skilled writer.

Given the painting metadata below, generate a **rich, structured, and beautifully formatted Markdown** explanation with:
- 🎨 Title and artist as a heading
- 🖼️ A short paragraph on what it represents
- 🕰️ When and why it was painted (if known)
- 🌍 Cultural or historical context

**Return valid Markdown only.**

Metadata:
- Title: {results['metadata']['title']}
- Artist: {results['metadata']['artist']}
- Style: {results['metadata']['style']}
- Genre: {results['metadata']['genre']}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content


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
st.markdown("<div class='subtitle'>Upload a painting and we'll tell you its story</div>", unsafe_allow_html=True)
st.markdown("")

uploaded_file = st.file_uploader("📤 Upload a painting", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])  # Wider column for best match

    with col1:
        st.image(image, caption="🖼️ Uploaded Painting", use_container_width=True)

    with col2:
        with st.spinner("🔍 Searching the model..."):

            embedding = get_image_embedding(image)

            query_response = index.query(
                vector=embedding.tolist(),
                top_k=1,
                include_metadata=True,
                namespace="ns1"
            )

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
                explanation = get_art_explanation(result)
        else:
            st.error("❌ Sorry, no match was found.")
            explanation = None

        if explanation:
            st.divider()
            st.markdown(explanation, unsafe_allow_html=True)