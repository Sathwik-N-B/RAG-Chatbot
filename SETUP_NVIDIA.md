# 🌐 NVIDIA z-ai Integration Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get NVIDIA API Key
1. Visit [https://build.nvidia.com/](https://build.nvidia.com/)
2. Sign up or log in to your NVIDIA account
3. Click "Create new secret key"
4. Copy the key (save it somewhere safe)

### 3. Configure `.env` File
Create or update your `.env` file:

```env
# NVIDIA Configuration
NVIDIA_API_KEY=nvapi-your-actual-api-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=z-ai/glm-5.1

# Optional: Keep Ollama config as fallback
OLLAMA_API_URL=http://127.0.0.1:11434
MODEL=qwen2.5:1.5b
```

### 4. Run the App
```bash
streamlit run app.py
```

### 5. Select LLM Provider
In the sidebar, you'll see a radio button to choose between:
- **NVIDIA** ✅ (Requires API key)
- **Ollama** 📦 (Local, free)

---

## Model Selection

### z-ai/glm-5.1 (Recommended)
- **Cost**: Depends on NVIDIA Build pricing
- **Speed**: Fast responses
- **Use Case**: Document Q&A and reasoning
- **Set in `.env`**: `NVIDIA_MODEL=z-ai/glm-5.1`

---

## Features

### ✅ Hybrid RAG with NVIDIA
- Uses your documents for context (no tokens wasted on raw docs)
- NVIDIA only processes the refined context + your question
- GraphRAG knowledge graphs for better understanding
- Neural reranking for top-quality results

### ✅ HyDE Query Expansion
- When enabled, also uses NVIDIA to expand your queries
- Better retrieval through hypothetical answer generation

### ✅ Real-time Streaming
- Responses stream in real-time as they're generated
- See tokens appearing instantly

---

## Cost Estimation

**Example: Typical RAG Query**

1. Upload a 10-page PDF (~5,000 tokens)
2. Ask a question (~50 tokens)
3. Retrieve context (~1,000 tokens)
4. Generate answer (~500 tokens output)

**Estimated cost per query:**
- **GPT-3.5-turbo**: ~$0.003-0.005
- **GPT-4**: ~$0.015-0.025

**Monthly budget examples (1000 queries):**
- GPT-3.5-turbo: ~$3-5
- GPT-4: ~$15-25

---

## Troubleshooting

### ❌ "NVIDIA API key not found"
- Check your `.env` file has `NVIDIA_API_KEY=nvapi-...`
- Restart the app: `streamlit run app.py`
- Verify the key is valid in NVIDIA Build

### ❌ "Rate limit exceeded"
- You've hit NVIDIA rate limiting
- Wait a few moments before sending more queries
- Consider checking your NVIDIA Build limits

### ❌ "Model not found: z-ai/glm-5.1"
- Your API key doesn't have access to that model
- Confirm the model name in NVIDIA Build

### ❌ "Connection timeout"
- NVIDIA API servers might be temporarily unavailable
- Check NVIDIA Build status or try again later
- Try again in a few moments

---

## Switching Between Providers

### At Runtime (Easiest)
Use the sidebar radio button to switch between NVIDIA and Ollama instantly

### Change Default
Edit your `.env` file and remove `NVIDIA_API_KEY` to default to Ollama

### Completely Disable NVIDIA
Leave `NVIDIA_API_KEY` empty in `.env`

---

## Security Tips

⚠️ **IMPORTANT**: Never share or commit your API key!

```bash
# Add to .gitignore (should already be there)
.env
.env.local
```

Never do this:
```bash
# ❌ DON'T
export NVIDIA_API_KEY=nvapi-xxx-in-github-yyy
# ❌ DON'T
git add .env
```

---

## Want to Contribute?

Have ideas for improvements? Found a bug? Open an issue or submit a PR!

Enjoy your RAG chatbot with NVIDIA z-ai! 🚀
