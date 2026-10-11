# PDF RAG Lab (free, beginner-friendly)

A simple Retrieval-Augmented Generation (RAG) teaching demo. Students upload a PDF, extract and chunk its text, index the chunks with TF-IDF, retrieve relevant passages, and generate an answer with Google's open-weight `google/flan-t5-small` model.

## Files
- `app.py` — Streamlit front end and RAG pipeline
- `requirements.txt` — Python dependencies

## Important notes
- No paid API, API key, or external inference endpoint is used.
- FLAN-T5-small is downloaded from Hugging Face on first run; its model card lists an Apache-2.0 license: https://huggingface.co/google/flan-t5-small
- "Indexing" is not model training: this demo does not update model weights. It creates a document-specific searchable index.
- There is no truly unlimited hosted free model guarantee. This app avoids per-token API fees by running inference in the app's available CPU/RAM, but the host has compute, memory, and usage limits.
- The demo works best with PDFs containing selectable text. Scanned/image-only PDFs need OCR first.
- Do not upload confidential or sensitive documents to a public deployment.

## Run locally
Use Python 3.10 or 3.11 for a straightforward setup.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model files and may take several minutes.

## Deploy from GitHub to Streamlit Community Cloud
1. Create/sign in to GitHub at https://github.com/.
2. Create a new repository, e.g. `pdf-rag-lab`.
3. Upload `app.py` and `requirements.txt` to the repository root and commit the files.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app** (or **Deploy an app**).
6. Select your repository, branch (usually `main`), and main file path `app.py`.
7. Choose a supported Python version in Advanced settings if needed, then deploy.
8. Wait for dependency installation and the first model download. The first startup may be slow.
9. Open the generated `*.streamlit.app` URL and upload a test PDF.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
Dependency guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Suggested classroom exercise
1. Ask students to explain the difference between chunking, retrieval, and generation.
2. Change `chunk_size` and `overlap` in `make_chunks()`.
3. Change the number of retrieved passages (`top_k`).
4. Ask one answerable and one unanswerable question; discuss grounding and hallucination.
5. Compare TF-IDF retrieval with an embedding-based retriever as an extension.
