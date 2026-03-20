import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Meeting Intelligence Assistant", layout="wide")

st.title("🎧 Meeting Intelligence Assistant")
st.write("Upload a meeting recording and ask questions to find exact moments.")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel("base")

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

whisper_model = load_whisper()
embed_model = load_embedder()

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    # Save uploaded file
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Processing audio... Please wait.")

    # -----------------------------
    # Transcription
    # -----------------------------
    with st.spinner("Transcribing audio..."):
        segments, _ = whisper_model.transcribe("temp_audio.mp3")

    chunks = []
    for seg in segments:
        chunks.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end
        })

    st.success("✅ Transcription complete!")

    # -----------------------------
    # Embeddings + FAISS
    # -----------------------------
    texts = [c["text"] for c in chunks]

    with st.spinner("Generating embeddings..."):
        embeddings = embed_model.encode(texts)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    st.success("✅ Ready! Ask your question below.")

    # -----------------------------
    # Search function
    # -----------------------------
    def search(query, k=3):
        q_emb = embed_model.encode([query])
        D, I = index.search(np.array(q_emb), k)
        return [chunks[i] for i in I[0]]

    # -----------------------------
    # Query input
    # -----------------------------
    query = st.text_input("🔍 Ask a question about the meeting:")

    if query:
        results = search(query)

        st.markdown("### 📌 Relevant Moments")

        for r in results:
            st.markdown(
                f"""
                **⏱ {r['start']:.2f}s - {r['end']:.2f}s**  
                {r['text']}
                """
            )
