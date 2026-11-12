# FinCon RAG Lab — John Khamis

**Project owner: [John Khamis](https://github.com/Yassahany2004)**

A Retrieval-Augmented Generation (RAG) application built with Streamlit.

**Live application:** Add the URL here after deploying the new repository to Streamlit Community Cloud.

![Application interface showing an answer and its source](reports/preview-app.png)

## Project Status

Groq has been enabled and live generation has been tested with PRF, HyDE, and Multi-Query. The application can also generate answers in Arabic with source references.

- Generation report: `reports/groq-live-report.json`
- Retrieval, comparison, file-upload, and benchmark report: `reports/live-deployment-report.json`

[Deploy this repository to Streamlit](https://share.streamlit.io/deploy?repository=Yassahany2004/fincon-rag-task&branch=main&mainModule=streamlit_app.py)

## Implemented Features

| Stage | Reference approach | Implemented approach |
| --- | --- | --- |
| Document chunking | Fixed-character Parent/Child chunking | Sentence windows capped at 80 words, or splitting based on lexical cohesion |
| Query expansion | spaCy and a financial dictionary | Pseudo-Relevance Feedback (PRF), Rocchio, Multi-Query, and HyDE |
| Query routing | No router was found in the supplied code | Domain-centroid similarity or KNN voting |
| Retrieval | MiniLM + ChromaDB + BM25 | BM25L + local LSA vector search |
| Fusion | Reciprocal Rank Fusion (RRF) | Weighted relative scores, with RRF available for comparison |
| Reranking | BGE Cross-Encoder | Maximal Marginal Relevance (MMR) or relevance ordering |
| Context | Parent-chunk retrieval | Sentence-extraction compression or neighboring windows |
| Generation | Groq with source-reference instructions | Groq with reference-ID validation and snippet fallback when validation fails |
| Evaluation | Original evaluation file | 14 questions with record-level labels and a comparison of four configurations |

## Run Locally

**Requirement:** Python 3.12

Create and activate a virtual environment.

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the dependencies and start the application:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The application does not require FastAPI, Elasticsearch, Docker, a GPU, or downloading models from Hugging Face. Indexing is performed automatically in memory.

## Enable Answer Generation

The application works without an API key and displays original source sentences with references. This is retrieval-and-extraction mode only. To enable generative RAG, configure Groq.

1. Create an API key at [Groq Console](https://console.groq.com/keys).
2. Enter the key in the **Answer generation** section of the application, or configure it locally as described below.
3. For Streamlit Community Cloud, add the key through the app's **Secrets** settings.

For local use, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add:

```toml
GROQ_API_KEY = "your-own-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

The model name can be changed if the list of models available to your Groq account changes. Previous settings for Llama 3.1 8B and Llama 3.3 70B are automatically mapped to GPT-OSS 20B because those models are no longer available to regular accounts. You do not need to change your API key or recreate your secrets.

Generation uses `reasoning_effort=low` with a 4096-token limit. **Never commit your secrets file to GitHub.** When generation is enabled, the questions and selected passages are sent to Groq.

## Deploy to Streamlit Community Cloud

1. Visit [Streamlit Community Cloud](https://share.streamlit.io) and sign in with GitHub. Complete Streamlit account setup if prompted.
2. Connect your GitHub account and grant Streamlit access to this repository.
3. Click **Create app**, then **Yup, I have an app**.
4. Select repository `Yassahany2004/fincon-rag-task` and branch `main`.
5. Set **Main file path** to `streamlit_app.py`.
6. In **Advanced settings**, select Python 3.12. Add the Groq settings above under **Secrets** to enable answer generation.
7. Click **Deploy**. Once deployment succeeds, copy the generated `streamlit.app` URL.

The application is ready to be deployed, but having the code on GitHub does not mean it has already been deployed to Streamlit. Creating the account, connecting the repository, and obtaining the application URL are separate steps.

[Official Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## How to Test the Application

1. Ask a Form 8-K question in the interface and open **Evidence trail**.
2. In the sidebar, switch from PRF to Rocchio. Ask the question again and open **Inspect the pipeline**.
3. Try **Centroid** and then **KNN**. Observe the domain scores. The router applies a slight preference but does not remove the other domains.
4. Use **Compare methods** with the same question to compare retrieved sources and their ranking.
5. Click **Run retrieval benchmark** and save the report. The actual results are also available in `reports/evaluation-report.json`.
6. Enable Groq to test generation, Multi-Query, and HyDE. Without an API key, the application explicitly reports that it is falling back to PRF.
7. Upload a text document or PDF and select **Uploads only** to search only within your uploaded file.

## Tests and Evaluation

Install the development dependencies, run the tests, and execute the evaluation module:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m finrag.evaluation
```

The tests cover:

- Correct source retrieval
- Questions outside the supported domain
- Document-group isolation
- Groq failures
- Rejection of nonexistent source references
- Preventing HyDE output from being used as evidence
- Streamlit interface startup, answering, and comparison

Local tests use mocked responses and do not require an API key. Live Groq generation was also tested on the deployed version, including HyDE, Multi-Query, and Arabic answer generation.

**Test results:** 38 tests passed. In addition, Chromium tests covered search, comparison, benchmarking, file upload and isolated file search, session isolation, and mobile rendering. The browser summary is available in `reports/browser-report.json`.

GitHub Actions runs the tests and evaluation on every push and pull request. The interface benchmark always uses the demo dataset, not user-uploaded files.

## Metrics and Limitations

- **Hit@4:** Whether the correct record appears among the final retrieved results, up to four passages.
- **MRR@4:** The reciprocal rank of the correct record among retrieved records, after duplicate IDs are removed.
- **Evidence-term coverage:** The proportion of specified terms from the test question that appear in the final context. This is not Context Recall or Answer Correctness.
- **Retrieval time:** Does not include indexing or answer generation. It is measured locally and is not a Streamlit Cloud performance measurement.

The evaluation dataset is small and domain-specific; it is not an independent generalization test. We do not assume that every change improves retrieval results. Reference validation checks that reference IDs exist and that passages are covered; it does not prove that every statement generated by the model is correct.

## Technical Notes

- **LSA:** Implemented using TF-IDF, followed by Truncated SVD, vector normalization, and cosine-similarity search. It is not a pretrained neural embedding model.
- **Semantic chunking:** The implementation uses TF-IDF similarity between adjacent sentences. The name does not mean that neural semantic embeddings are used.
- **PRF:** Adds six terms based on the average of the top three TF-IDF results.
- **Rocchio:** Uses `0.8 × query + 0.2 × positive-centroid` in dense-vector space, without negative examples.
- **Query router:** Applies a boost of `1.15` when the domain is clear; otherwise, it searches all domains.
- **MMR:** Balances the relevance of a result to the query against its similarity to already selected results.
- **Arabic support:** Local Arabic query support uses a small financial dictionary, not general-purpose translation. Groq can generate Arabic answers when enabled. Without Groq, retrieved passages remain in the document's original language.

## File Uploads and Limitations

Supported formats:

- Text-based PDF
- TXT
- Markdown (MD)
- JSON containing `content`, `text`, or `response`

Upload limits:

- Maximum file size: 10 MB
- Maximum number of files: 20
- Maximum PDF length: 150 pages
- Maximum combined collection size: 1,500,000 characters

Scanned PDFs require OCR outside the application. User-uploaded indexes are stored in Streamlit Session State, while only the demo index is shared read-only. Uploaded files are not stored permanently; restarting the server clears the session.

## Project Structure

```text
streamlit_app.py
README.md
OWNER.md
UPLOAD_GUIDE_AR.md
requirements.txt
requirements-dev.txt
pytest.ini
.gitignore
finrag/
  __init__.py
  documents.py
  engine.py
  generation.py
  evaluation.py
data/
  baai_financial_corpus.json
  financial_regulatory_standards_2026.json
  benchmark.json
reports/
  browser-report.json
  evaluation-report.json
  groq-live-report.json
  live-deployment-report.json
  preview-app.png
tests/
  test_pipeline.py
  test_streamlit.py
.streamlit/
  config.toml
  secrets.toml.example
.github/
  workflows/
    ci.yml
```
