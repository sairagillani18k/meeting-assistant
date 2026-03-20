import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Meeting Intelligence Assistant", layout="wide")

st.title("🎧 Meeting Intelligence Assistant")
st.write("Upload a meeting recording and ask questions to find exact moments.")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

whisper_model = load_whisper()
embed_model = load_embedder()

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("⏳ Transcribing audio..."):
        result = whisper_model.transcribe("temp_audio.mp3", fp16=False)

    chunks = []
    for seg in result["segments"]:
        chunks.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"]
        })

    st.success("✅ Transcription complete!")

    # -----------------------------
    # Embeddings + FAISS
    # -----------------------------
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    st.success("✅ Ready! Ask your question below.")

    # -----------------------------
    # Search
    # -----------------------------
    def search(query, k=3):
        q_emb = embed_model.encode([query])
        D, I = index.search(np.array(q_emb), k)
        return [chunks[i] for i in I[0]]

    # -----------------------------
    # Query UI
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