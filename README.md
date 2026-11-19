# DL & GenAI Workspace

My notes and code for deep learning, transformers, LLMs, RAG and agents.
Classical ML is in [Machine-learning-workspace](https://github.com/vivekdongare81/Machine-learning-workspace).

Each notebook starts with a contents list and ends with interview questions.

## Roadmap

| Folder | Topics |
|---|---|
| 01-neural-networks | neurons, activations, loss functions, backpropagation and training, all from scratch in NumPy |
| 02-pytorch-basics | tensors, autograd, `nn.Module`, DataLoader, the training loop, GPU / MPS, saving models |
| 03-training-deep-networks | optimizers (SGD, momentum, Adam), learning-rate schedules, initialization, batch and layer norm, dropout, residual connections |
| 04-cnn-and-transfer-learning | convolutions, pooling, a small CNN, fine-tuning a pretrained model |
| 05-sequence-models | RNN, LSTM, GRU, and why attention replaced them |
| 06-transformers | tokenization (BPE), embeddings, self-attention from scratch, multi-head attention, positional encoding, a tiny GPT |
| 07-llms | pre-training, SFT, RLHF and DPO, decoding (temperature, top-k, top-p), KV cache, quantization, Hugging Face models |
| 08-llm-apis-and-prompting | calling LLM APIs, prompting, structured output, tool calling, streaming, prompt caching |
| 09-embeddings-and-vector-search | sentence embeddings, vector databases, ANN indexes, hybrid search |
| 10-rag | chunking, retrieval, reranking, evaluating RAG |
| 11-agents | a tool-calling loop from scratch, ReAct, memory, MCP, multi-agent patterns |
| 12-llm-evaluation | golden datasets, LLM-as-a-judge, regression tests for prompts |
| 13-fine-tuning | LoRA, QLoRA, preparing data, evaluating the result |
| 14-llmops-and-serving | serving models, streaming APIs, caching, cost, monitoring, guardrails |
| 15-projects | end-to-end GenAI projects |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name dl-genai --display-name "dl-genai (.venv)"
```

Then pick the `dl-genai (.venv)` kernel in Jupyter or VS Code.
