import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚", layout="wide")

MODEL_NAME = "google/flan-t5-small"

st.title("📚 PDF Question Answering — RAG Lab")
st.caption("A teaching demo: upload a PDF, index its text, retrieve relevant chunks, and generate an answer.")

with st.expander("What is happening behind the scenes?"):
    st.markdown("""
    1. **Load:** extract text from the uploaded PDF.
    2. **Chunk:** split the text into overlapping passages.
    3. **Index:** build a TF-IDF search index over the passages.
    4. **Retrieve:** find the passages most relevant to a question.
    5. **Generate:** ask the open model FLAN-T5-small to answer using those passages.

    This demo does **not fine-tune/train model weights**. It builds a document-specific retrieval index, which is the usual setup for a basic RAG system.
    """)

@st.cache_resource(show_spinner="Loading the free model (first run may take a few minutes)...")
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))
    return pages

def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for page_number, text in pages:
        cleaned = " ".join(text.split())
        start = 0
        while start < len(cleaned):
            piece = cleaned[start:start + chunk_size].strip()
            if piece:
                chunks.append({"page": page_number, "text": piece})
            if start + chunk_size >= len(cleaned):
                break
            start += chunk_size - overlap
    return chunks

def answer_question(question, context, tokenizer, model):
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you cannot find it in the PDF. "
        "Keep the answer clear and concise.\n\n"
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=2,
            do_sample=False
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_pdf:
    try:
        pages = extract_pdf_text(uploaded_pdf)
        if not pages:
            st.error("No selectable text was found. This may be a scanned PDF; run OCR on it first.")
            st.stop()

        chunks = make_chunks(pages)
        if not chunks:
            st.error("No text chunks could be created from this PDF.")
            st.stop()

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

        st.success(f"Document indexed: {len(pages)} text page(s), {len(chunks)} chunk(s).")
        st.info("Your document index is ready. Ask a question below.")

        with st.sidebar:
            st.header("RAG settings")
            top_k = st.slider("Passages to retrieve", min_value=1, max_value=min(5, len(chunks)), value=min(3, len(chunks)))
            st.write(f"Generator: `{MODEL_NAME}`")
            st.write("Retrieval: TF-IDF (local, no API key)")

        question = st.text_input("Ask a question about your PDF", placeholder="What are the main ideas in this document?")
        if st.button("Get answer", type="primary", disabled=not question.strip()):
            query_vector = vectorizer.transform([question])
            scores = cosine_similarity(query_vector, matrix).flatten()
            ranked = scores.argsort()[::-1][:top_k]
            retrieved = [chunks[i] for i in ranked if scores[i] > 0]

            if not retrieved:
                st.warning("I could not find a matching passage. Try rephrasing your question.")
            else:
                context = "\n\n".join(
                    f"[Page {item['page']}] {item['text']}" for item in retrieved
                )
                tokenizer, model = load_generator()
                with st.spinner("Generating an answer from the retrieved passages..."):
                    answer = answer_question(question, context, tokenizer, model)

                st.subheader("Answer")
                st.write(answer)
                st.subheader("Retrieved source passages")
                for item in retrieved:
                    with st.expander(f"PDF page {item['page']}"):
                        st.write(item["text"])

        with st.expander("Show indexed chunks (useful for teaching/debugging)"):
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i} — PDF page {chunk['page']}**")
                st.write(chunk["text"])
    except Exception as exc:
        st.error("The PDF could not be processed. Try another text-based PDF.")
        st.exception(exc)
else:
    st.markdown("**Start here:** upload a text-based PDF using the uploader above.")
    st.markdown("No paid API key is required. The first model load downloads model files and may take time.")
