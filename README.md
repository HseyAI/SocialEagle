# 🧪 RAG Learning Lab - Interactive Educational Tool

> **An interactive platform to learn, experiment, and understand Retrieval-Augmented Generation (RAG) concepts through hands-on experimentation.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📖 Table of Contents
- [What is RAG Learning Lab?](#-what-is-rag-learning-lab)
- [Why This Tool?](#-why-this-tool)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Installation Guide](#-installation-guide)
- [Usage & Experiments](#-usage--experiments)
- [Configuration Options](#-configuration-options)
- [Technical Architecture](#-technical-architecture)
- [Educational Use Cases](#-educational-use-cases)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 What is RAG Learning Lab?

**RAG Learning Lab** is an educational tool designed to demystify **Retrieval-Augmented Generation (RAG)** - a technique that combines information retrieval with AI language models to provide accurate, context-aware responses.

Unlike traditional tutorials, this tool lets you **see, touch, and experiment** with every component of a RAG system in real-time.

### 🌟 Perfect For:
- 🎓 **Students** learning about AI and NLP
- 👨‍💻 **Developers** exploring RAG implementations
- 👩‍🏫 **Educators** teaching information retrieval concepts
- 🔬 **Researchers** experimenting with different RAG configurations
- 📊 **Product Managers** understanding RAG capabilities and limitations

---

## 💡 Why This Tool?

### The Problem
Most RAG tutorials show you the **final result** but hide the **process**. You don't see:
- How text gets split into chunks
- Why chunk size matters
- What embeddings actually do
- How retrieval works under the hood
- The difference between RAG with and without AI

### The Solution
**RAG Learning Lab** makes the invisible visible:
- ✅ **See** how your documents are processed in real-time
- ✅ **Experiment** with different settings and immediately see results
- ✅ **Compare** pure RAG vs AI-enhanced responses
- ✅ **Understand** each component through interactive tooltips
- ✅ **Learn** through hands-on experimentation, not passive reading

---

## ✨ Key Features

### 1. 📊 **Interactive Text Processing**
- **Adjustable Chunk Sizes** (200-2000 characters)
  - See how different sizes affect retrieval quality
  - Understand the trade-off between precision and context
- **Configurable Overlap** (0-500 characters)
  - Prevent important information from being split
  - Visualize how chunks overlap
- **Sample Chunk Viewer**
  - Inspect exactly how your text was split
  - See chunk boundaries in real-time

### 2. 🔢 **Multiple Embedding Models**
Choose from three embedding models and see how they differ:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| **all-MiniLM-L6-v2** | ⚡⚡⚡ Fast | Good | Learning & Testing |
| **all-mpnet-base-v2** | ⚡⚡ Medium | Excellent | Production Use |
| **paraphrase-multilingual-MiniLM-L12-v2** | ⚡⚡ Medium | Good | Multiple Languages |

### 3. 🔍 **Transparent Retrieval System**
- **Adjustable Top-K Results** (1-10 chunks)
  - Control how many relevant chunks to retrieve
  - See relevance scores and rankings
- **Source Citations**
  - Every result shows source file and page number
  - Trace answers back to original documents
- **Visual Similarity**
  - Understand why certain chunks were retrieved

### 4. 🤖 **Optional AI Integration**
**Unique Feature:** Toggle AI on/off to see the difference!

#### Without AI (Pure RAG):
- Shows raw retrieved chunks
- Perfect for understanding retrieval mechanics
- See exactly what the vector database found

#### With AI (RAG + Generation):
- Human-readable answers generated from chunks
- Choose from multiple providers:
  - 🟢 **OpenAI GPT** (Best quality, $5 free credit)
  - 🔵 **Hugging Face** (100% free, open-source)
  - 🔴 **Google Gemini** (Fast, free tier available)

### 5. 🎨 **Creativity Control**
When AI is enabled, control the temperature (0.0 - 1.0):
- **0.0 - 0.3:** 📚 Strict (only uses provided context, minimal hallucination)
- **0.4 - 0.6:** ⚖️ Balanced (creative but grounded)
- **0.7 - 1.0:** 🎨 Creative (may add information, risk of hallucination)

**Learn about hallucination** by setting creativity high and asking questions not in your documents!

### 6. 💾 **Persistent Vector Database**
- Uses **ChromaDB** for efficient vector storage
- Data persists between sessions
- Reset functionality to start fresh

### 7. 📚 **Multi-Document Support**
- Upload multiple PDFs simultaneously
- Process documents of any size
- Track which files have been processed

### 8. 🎓 **Educational Tooltips**
Every setting includes:
- Clear explanation of what it does
- Why it matters
- Example scenarios
- Live feedback showing current values

---

## 🔄 How It Works

### The RAG Pipeline (3 Steps)

```
┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: INDEXING                         │
│  📄 PDF Upload → 📝 Text Extraction → ✂️ Chunking →        │
│  🔢 Embeddings → 💾 Vector Database                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STEP 2: RETRIEVAL                        │
│  ❓ User Question → 🔢 Question Embedding →                 │
│  🔍 Similarity Search → 📊 Top-K Chunks                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STEP 3: GENERATION                       │
│  📋 Retrieved Chunks → 🤖 AI (Optional) →                   │
│  💬 Final Answer with Sources                               │
└─────────────────────────────────────────────────────────────┘
```

### What Makes This Tool Special?

1. **Transparency:** See each step happening in real-time
2. **Flexibility:** Turn AI on/off to understand pure RAG
3. **Experimentation:** Change settings and immediately see impact
4. **Education:** Learn by doing, not just reading

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.11 or higher
- 2GB free disk space (for models)
- Windows/Mac/Linux

### Step-by-Step Installation

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/HseyAI/SocialEagle.git
cd SocialEagle
```

#### 2️⃣ Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ (Optional) Setup AI Keys
If you want to use AI features, create `.streamlit/secrets.toml`:

```toml
# OpenAI (Get $5 free credit at platform.openai.com/api-keys)
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxx"

# Hugging Face (Free token at huggingface.co/settings/tokens)
HF_API_KEY = "hf_xxxxxxxxxxxxx"

# Google Gemini (Free at aistudio.google.com/apikey)
GOOGLE_API_KEY = "AIzaxxxxxxxxxxxxx"
```

**Note:** You can skip this and use Pure RAG mode without any API keys!

#### 5️⃣ Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧪 Usage & Experiments

### Experiment 1: Understanding Pure RAG
**Goal:** See how retrieval works without AI

1. **Disable AI** (uncheck "Enable AI Assistant")
2. Upload a PDF textbook (e.g., science textbook)
3. Ask: *"What is photosynthesis?"*
4. **Observe:** You'll see raw chunks containing the answer
5. **Learn:** This is pure retrieval - no AI generation!

### Experiment 2: Impact of Chunk Size
**Goal:** Understand the chunk size vs context trade-off

1. Set **Chunk Size = 200**
2. Upload a document and ask a question
3. Note the results
4. **Reset memory** and change **Chunk Size = 2000**
5. Upload same document, ask same question
6. **Compare:** Small chunks = precise but fragmented, Large chunks = more context but less precise

### Experiment 3: AI Hallucination
**Goal:** See when and why AI hallucinates

1. Enable AI, set **Creativity = 0.9** (high)
2. Upload a history textbook
3. Ask something NOT in the book: *"What happened in the year 2050?"*
4. **Observe:** AI might make things up (hallucinate)
5. Now set **Creativity = 0.1** (low) and ask again
6. **Learn:** Lower creativity = more factual, higher = more creative but risky

### Experiment 4: Comparing Embedding Models
**Goal:** See how different models affect retrieval

1. Use **all-MiniLM-L6-v2**, upload doc, ask question
2. Note which chunks are retrieved
3. Reset, switch to **all-mpnet-base-v2**
4. Ask same question
5. **Compare:** Better models find more relevant chunks

### Experiment 5: Optimal Top-K
**Goal:** Find the sweet spot for number of chunks

1. Set **Top-K = 1**, ask a complex question
2. Increase to **Top-K = 5**, ask again
3. Try **Top-K = 10**
4. **Learn:** Too few = missing context, Too many = irrelevant noise

---

## ⚙️ Configuration Options

### 📊 Text Splitting Settings

#### Chunk Size
```
Range: 200 - 2000 characters
Default: 800 characters (~160 words)
```
**What it does:** Determines how large each text piece will be.

**When to adjust:**
- **Smaller (200-500):** For precise, specific queries (e.g., "What is the capital of France?")
- **Medium (600-1000):** Balanced - works for most use cases
- **Larger (1200-2000):** For questions needing broad context (e.g., "Explain the causes of World War II")

#### Overlap
```
Range: 0 - 500 characters
Default: 100 characters
```
**What it does:** How much text is shared between consecutive chunks.

**Why it matters:** Prevents splitting sentences/paragraphs awkwardly.

**Recommended:** 10-15% of chunk size

### 🔢 Embedding Models

#### all-MiniLM-L6-v2
- **Size:** 80MB
- **Speed:** Very Fast (⚡⚡⚡)
- **Quality:** Good
- **Use case:** Learning, quick tests, prototypes

#### all-mpnet-base-v2
- **Size:** 420MB
- **Speed:** Medium (⚡⚡)
- **Quality:** Excellent
- **Use case:** Production applications

#### paraphrase-multilingual-MiniLM-L12-v2
- **Size:** 470MB
- **Speed:** Medium (⚡⚡)
- **Quality:** Good
- **Languages:** 50+ languages
- **Use case:** Non-English documents

### 🔍 Retrieval Settings

#### Top-K (Number of Chunks)
```
Range: 1 - 10
Default: 4
```
**What it does:** How many relevant chunks to retrieve.

**Guidelines:**
- **K=1-2:** Simple factual questions
- **K=3-5:** Balanced (recommended)
- **K=6-10:** Complex questions needing multiple sources

### 🤖 AI Settings

#### Creativity Level (Temperature)
```
Range: 0.0 - 1.0
Default: 0.4
```

| Value | Behavior | Risk | Best For |
|-------|----------|------|----------|
| 0.0-0.2 | Very strict, factual | Low hallucination | Academic, legal documents |
| 0.3-0.5 | Balanced | Medium | General use |
| 0.6-0.8 | Creative, engaging | Higher hallucination | Creative writing, brainstorming |
| 0.9-1.0 | Very creative | High hallucination | Experimental only |

---

## 🏗️ Technical Architecture

### Tech Stack

```yaml
Frontend:
  - Streamlit (UI Framework)
  
Vector Database:
  - ChromaDB (Persistent Storage)
  
Embeddings:
  - Sentence-Transformers
  - Models: MiniLM, MPNet
  
Document Processing:
  - PyPDF (PDF Text Extraction)
  
AI Integration (Optional):
  - OpenAI GPT-3.5/4
  - Hugging Face Inference API
  - Google Gemini API
```

### System Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit Frontend                │
│  (Interactive UI, Settings, Chat)           │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          RAG System Core                    │
│                                             │
│  ┌─────────────┐    ┌──────────────┐      │
│  │   Document  │    │  Embedding   │      │
│  │  Processor  │───▶│    Model     │      │
│  └─────────────┘    └──────────────┘      │
│                             │               │
│                             ▼               │
│                    ┌──────────────┐        │
│                    │   ChromaDB   │        │
│                    │  Vector DB   │        │
│                    └──────────────┘        │
│                             │               │
│                             ▼               │
│                    ┌──────────────┐        │
│                    │  Retrieval   │        │
│                    │    Engine    │        │
│                    └──────────────┘        │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────▼────────┐
        │  AI Integration  │
        │    (Optional)    │
        └──────────────────┘
```

### Data Flow

1. **Upload:** User uploads PDF(s)
2. **Extract:** PyPDF extracts text from pages
3. **Chunk:** Text split into configurable chunks with overlap
4. **Embed:** Sentence-Transformer converts chunks to vectors (embeddings)
5. **Store:** ChromaDB stores vectors with metadata
6. **Query:** User asks question → converted to embedding
7. **Retrieve:** Vector similarity search finds top-K chunks
8. **Generate:** (If AI enabled) LLM generates answer from chunks
9. **Display:** Show results with source citations

---

## 🎓 Educational Use Cases

### 1. Computer Science Courses
**Topics:** Information Retrieval, NLP, Vector Databases

**Activities:**
- Compare different embedding models
- Analyze retrieval precision vs recall
- Study the impact of hyperparameters

### 2. AI/ML Workshops
**Topics:** RAG Architecture, Prompt Engineering

**Activities:**
- Build understanding of RAG pipeline
- Experiment with prompt templates
- Learn about AI hallucination

### 3. Research Projects
**Topics:** Document Analysis, Question Answering

**Activities:**
- Test different chunking strategies
- Evaluate retrieval quality
- Benchmark AI models

### 4. Self-Learning
**Topics:** AI Fundamentals, Practical NLP

**Activities:**
- Follow experiment guides
- Build intuition through hands-on practice
- Understand trade-offs in system design

---

## 🐛 Troubleshooting

### Issue: Model Download is Slow
**Solution:** First-time model downloads can take 5-10 minutes depending on internet speed. Subsequent runs are instant.

### Issue: "Import google.genai could not be resolved"
**Solution:** 
```bash
pip uninstall google-generativeai google-genai
pip install google-genai
```

### Issue: ChromaDB Permission Error
**Solution:** Delete `chroma_db_data` folder and restart the app.

### Issue: Out of Memory
**Solution:** 
- Use smaller embedding model (all-MiniLM-L6-v2)
- Reduce chunk size
- Process fewer documents at once

### Issue: AI API Quota Exceeded
**Solution:** 
- OpenAI: Check usage at platform.openai.com/usage
- Hugging Face: Wait for rate limit reset (usually 1 hour)
- Google: Switch to different model or wait 24 hours

### Issue: Slow Retrieval
**Solution:**
- Reduce Top-K value
- Use faster embedding model
- Check if vector DB is too large (reset if needed)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs via GitHub Issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add new experiment templates
- 🎨 Enhance UI/UX

### Development Setup
```bash
git clone https://github.com/HseyAI/SocialEagle.git
cd SocialEagle
pip install -r requirements.txt
# Make your changes
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💼 Author

**Yeshwanth**
- Role: Project Manager & Scrum Master
- Company: Noire Infini (Product Development)
- Client: Reckitt Canada (Enfamil.ca eCommerce)
- Location: Guindy, Chennai

### Connect
- GitHub: [@HseyAI](https://github.com/HseyAI)
- LinkedIn: [Add your LinkedIn]
- Email: [Add your email]

---

## 🙏 Acknowledgments

- **Sentence-Transformers** team for excellent embedding models
- **ChromaDB** for the vector database
- **Streamlit** for the amazing UI framework
- **OpenAI, Hugging Face, Google** for AI APIs

---

## 📊 Project Stats

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Streamlit Version](https://img.shields.io/badge/streamlit-1.52.2-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Roadmap

- [ ] Add support for more document formats (DOCX, TXT, MD)
- [ ] Implement query history and analytics
- [ ] Add visualization of embedding space
- [ ] Support for custom embedding models
- [ ] Batch processing of multiple queries
- [ ] Export results to PDF/CSV
- [ ] Add pre-built experiment templates
- [ ] Multi-language UI support

---

**⭐ If you find this tool helpful, please star the repository!**

**🐛 Found a bug? Open an issue!**

**💡 Have a feature idea? Start a discussion!**

---

*Built with ❤️ for education and learning*