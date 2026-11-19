# DL & GenAI Workspace

My notes and code for deep learning, transformers, LLMs, RAG and agents.
Classical ML is in [Machine-learning-workspace](https://github.com/vivekdongare81/Machine-learning-workspace).

Each notebook starts with a contents list and ends with the trade-offs and interview questions for its topic.

## Roadmap

### Deep learning

| Folder | Topics |
|---|---|
| 01-neural-networks | neurons, activations, loss functions, backpropagation, initialization, training from scratch in NumPy |
| 02-pytorch-basics | from NumPy arrays to tensors (NumPy and pandas are in the ML repo's `00-numpy-and-pandas`), tensors, autograd, `nn.Module`, Dataset and DataLoader, the training loop, GPU / MPS, saving and loading checkpoints, turning a notebook into a training script (config, logging, command-line arguments) |
| 03-training-deep-networks | optimizers (SGD, momentum, Adam, AdamW), learning-rate schedules and warm-up, batch and layer norm, dropout, weight decay, early stopping, residual connections, gradient clipping, mixed precision, gradient accumulation, experiment tracking (MLflow, Weights & Biases), multi-GPU training (DDP, FSDP) overview, debugging a training run |
| 04-cnn-and-transfer-learning | convolutions, pooling, a small CNN, transfer learning (short) |
| 05-sequence-models | RNN, LSTM, GRU, vanishing gradients through time, why attention replaced them (short) |

### Transformers and LLMs

| Folder | Topics |
|---|---|
| 06-transformers | tokenization (BPE, WordPiece), embeddings, self-attention from scratch, multi-head attention, positional encoding (sinusoidal, RoPE), encoder vs decoder (BERT vs GPT), training a tiny GPT |
| 07-llms | pre-training, SFT, RLHF and DPO, scaling laws, decoding (temperature, top-k, top-p), KV cache, context length, mixture of experts, reasoning models, hallucinations, quantization, the Hugging Face Hub, running open models locally (Hugging Face, Ollama) |

### Building with LLMs

| Folder | Topics |
|---|---|
| 08-llm-apis-and-prompting | calling LLM APIs, prompt design, few-shot and chain-of-thought, structured output with Pydantic, tool calling, streaming, prompt caching, token counting and cost |
| 09-embeddings-and-vector-search | sentence embeddings, choosing an embedding model, vector databases (FAISS, Qdrant, Postgres + pgvector), ANN indexes, hybrid search, metadata filtering |
| 10-rag | document processing (PDFs, tables, OCR), ingestion pipelines and re-indexing (scheduled jobs with Airflow or Prefect), chunking, retrieval, reranking, query rewriting, citations, access control, evaluating RAG, frameworks (LlamaIndex, LangChain) and when to skip them |
| 11-agents | a tool-calling loop from scratch, ReAct, planning, short- and long-term memory, MCP servers and clients, multi-agent patterns, retries and failure handling, human approval steps, long-running agents, LangGraph and vendor agent SDKs, agent evals |

### Production

| Folder | Topics |
|---|---|
| 12-llm-evaluation | golden datasets, LLM-as-a-judge, RAG and agent metrics, prompt versioning, prompt regression tests in CI, online feedback and A/B tests |
| 13-fine-tuning | when to fine-tune, LoRA and QLoRA, preparing data, SFT vs DPO, distillation, evaluating the result (on a Colab or Kaggle GPU) |
| 14-llmops-and-serving | FastAPI streaming services, async Python, queues and background jobs, rate limiting, Redis caching (exact and semantic), serving open models (Ollama, vLLM), LLM gateways and fallback across providers (LiteLLM), cloud LLM services (AWS Bedrock, GCP Vertex AI, Azure OpenAI), Docker Compose, CI/CD with GitHub Actions, Kubernetes basics, cost and latency, tracing and monitoring (OpenTelemetry, Langfuse, Prometheus + Grafana) |
| 15-llm-security | prompt injection, jailbreaks, PII handling, guardrails, document permissions in RAG, safe tool use |
| 16-multimodal | vision-language models, extracting data from images and scanned documents, speech-to-text basics |
| 17-genai-system-design | design rounds: a RAG chatbot, a support agent, semantic search, a document extraction pipeline |

### Projects

| Folder | Topics |
|---|---|
| 18-projects | RAG assistant over a document set, an agent with tools over MCP, a fine-tuned small model for one task; each with tests, CI, Docker, evals and a README of design decisions and trade-offs |

## Everyday tools used in the projects

| Area | Tools |
|---|---|
| Python project setup | uv, `pyproject.toml`, `src/` layout, type hints |
| Code quality | Git, pytest, ruff, mypy, pre-commit, GitHub Actions |
| Config and data models | Pydantic, pydantic-settings, `.env` files |
| APIs and services | FastAPI, async Python, Docker, Docker Compose |
| Data stores | Postgres + pgvector, Qdrant, Redis |
| Models | PyTorch, Hugging Face Hub and Transformers, sentence-transformers, Ollama, vLLM |
| LLM apps | the official LLM SDKs, LiteLLM, MCP, LangGraph |
| Experiments and evals | MLflow or Weights & Biases, eval datasets in the repo, LLM-as-a-judge |
| Monitoring | structured logging, OpenTelemetry, Langfuse, Prometheus + Grafana |
| Deployment | a cloud provider (AWS or GCP), container hosting, Kubernetes basics |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name dl-genai --display-name "dl-genai (.venv)"
```

Then pick the `dl-genai (.venv)` kernel in Jupyter or VS Code.
