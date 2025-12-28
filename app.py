import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import uuid

# --- CONFIGURATION ---
VECTOR_DB_PATH = "./chroma_db_data"

# --- UI STYLING ---
st.set_page_config(page_title="📚 RAG Learning Lab", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { 
        background-color: #262730; 
        color: white; 
    }
    .stChatMessage {
        background-color: #262730;
        border: 1px solid #444;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .info-box {
        background-color: #1a1d29;
        border-left: 4px solid #4CAF50;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC ---

class RAGSystem:
    def __init__(self, embedding_model):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="educational_rag")
        
        # Load embedding model based on selection
        if "embedder" not in st.session_state or st.session_state.get("current_model") != embedding_model:
            with st.spinner(f"🧠 Loading embedding model: {embedding_model}..."):
                st.session_state.embedder = SentenceTransformer(embedding_model)
                st.session_state.current_model = embedding_model
        
        self.embedder = st.session_state.embedder

    def ingest_files(self, uploaded_files, chunk_size, overlap):
        """Process and store documents in vector database"""
        documents = []
        metadatas = []
        ids = []
        
        progress_bar = st.progress(0, text="📖 Step 1: Reading PDF files...")
        total_files = len(uploaded_files)

        # Step 1: Extract text from PDFs
        for f_idx, file_obj in enumerate(uploaded_files):
            reader = PdfReader(file_obj)
            file_name = file_obj.name
            
            for p_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text: continue
                
                # Step 2: Split text into chunks
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    if end < len(text):
                        last_space = text.rfind(' ', start, end)
                        if last_space != -1: end = last_space
                    
                    chunk = text[start:end].strip()
                    if len(chunk) > 50:
                        documents.append(chunk)
                        metadatas.append({
                            "source": file_name, 
                            "page": p_idx + 1,
                            "chunk_size": chunk_size
                        })
                        ids.append(f"{uuid.uuid4().hex}")
                    
                    start = end - overlap
            
            progress_bar.progress((f_idx + 1) / total_files, 
                                 text=f"📄 Processed: {file_name}")

        if not documents:
            return 0, []

        # Step 3: Create embeddings and store
        progress_bar.progress(0.5, text="🔢 Step 2: Creating embeddings (converting text to numbers)...")
        
        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()
        
        progress_bar.progress(0.8, text="💾 Step 3: Storing in vector database...")
        
        # Store in ChromaDB
        batch_size = 256
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        
        progress_bar.empty()
        
        # Return sample chunks for display
        sample_chunks = documents[:3] if len(documents) >= 3 else documents
        return len(documents), sample_chunks

    def search(self, query, top_k):
        """Search for relevant chunks in vector database"""
        query_vec = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_vec, n_results=top_k)
        return results

    def reset_memory(self):
        """Clear all stored data"""
        self.client.delete_collection("educational_rag")
        self.collection = self.client.get_or_create_collection(name="educational_rag")

# --- MAIN APPLICATION ---

def main():
    # Header
    st.title("🧪 RAG Learning Lab")
    st.markdown("**Learn how Retrieval-Augmented Generation works by experimenting with different settings!**")
    
    # --- SIDEBAR: RAG CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ RAG Configuration")
        
        st.markdown("---")
        st.subheader("📊 1. Text Splitting Settings")
        
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
            help="💡 **What it does:** Breaks your document into smaller pieces of this size. Smaller = more precise but may lose context. Larger = more context but less precise."
        )
        st.caption(f"📏 Current: Each chunk will be ~{chunk_size} characters (~{chunk_size//5} words)")
        
        overlap = st.slider(
            "Overlap (characters)",
            min_value=0,
            max_value=500,
            value=100,
            step=50,
            help="💡 **What it does:** How much text overlaps between chunks. Prevents important information from being split awkwardly between chunks."
        )
        st.caption(f"🔗 Current: {overlap} characters overlap between chunks")
        
        st.markdown("---")
        st.subheader("🔢 2. Embedding Model")
        
        embedding_model = st.selectbox(
            "Choose Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            help="💡 **What it does:** Converts text into numbers (vectors) so the computer can understand similarity. Different models have different strengths."
        )
        
        model_info = {
            "all-MiniLM-L6-v2": "⚡ Fast & Lightweight (Good for learning)",
            "all-mpnet-base-v2": "🎯 More Accurate (Better quality)",
            "paraphrase-multilingual-MiniLM-L12-v2": "🌍 Multilingual (Works with multiple languages)"
        }
        st.caption(model_info[embedding_model])
        
        st.markdown("---")
        st.subheader("🔍 3. Retrieval Settings")
        
        top_k = st.slider(
            "Number of Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=4,
            help="💡 **What it does:** How many relevant chunks to find when you ask a question. More chunks = more context but may include irrelevant info."
        )
        st.caption(f"📚 Will retrieve top {top_k} most relevant chunks")
        
        st.markdown("---")
        st.subheader("🤖 4. AI Integration (Optional)")
        
        use_ai = st.checkbox(
            "Enable AI Assistant",
            value=False,
            help="💡 **What it does:** Uses AI to generate human-like answers from the retrieved chunks. Without AI, you'll see raw chunks (pure RAG)."
        )
        
        ai_key = None
        ai_provider = None
        creativity = 0.4
        
        if use_ai:
            ai_provider = st.selectbox(
                "AI Provider",
                options=["OpenAI (GPT)", "Hugging Face", "Google (Gemini)"],
                help="Choose which AI service to use"
            )
            
            if ai_provider == "OpenAI (GPT)":
                ai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Get free $5 credit at platform.openai.com"
                )
            elif ai_provider == "Hugging Face":
                ai_key = st.text_input(
                    "Hugging Face Token",
                    type="password",
                    help="Get free token at huggingface.co/settings/tokens"
                )
            elif ai_provider == "Google (Gemini)":
                ai_key = st.text_input(
                    "Google API Key",
                    type="password",
                    help="Get free key at aistudio.google.com"
                )
            
            if ai_key:
                creativity = st.slider(
                    "AI Creativity Level",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.1,
                    help="💡 **What it does:** Controls how creative vs factual the AI is. Low = strict to text, High = more creative but may hallucinate (make things up)."
                )
                st.caption(f"🎨 Current: {'Very Strict' if creativity < 0.3 else 'Balanced' if creativity < 0.7 else 'Creative (may hallucinate)'}")
        
        st.markdown("---")
        st.subheader("📚 5. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF textbooks or documents"
        )
        
        if "processed_files" in st.session_state and st.session_state.processed_files:
            st.success(f"✅ {len(st.session_state.processed_files)} files loaded")
            st.info(f"📦 {st.session_state.get('total_chunks', 0)} chunks stored in vector DB")
            
            if st.button("🗑️ Clear All Data"):
                rag = RAGSystem(embedding_model)
                rag.reset_memory()
                st.session_state.processed_files = set()
                st.session_state.total_chunks = 0
                st.session_state.sample_chunks = []
                st.rerun()

    # --- MAIN AREA ---
    
    # Initialize RAG system
    rag = RAGSystem(embedding_model)
    
    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
        st.session_state.total_chunks = 0
        st.session_state.sample_chunks = []
    
    # Process uploaded files
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            st.info("🔄 Processing new files...")
            total_chunks, sample_chunks = rag.ingest_files(new_files, chunk_size, overlap)
            
            for f in new_files:
                st.session_state.processed_files.add(f.name)
            
            st.session_state.total_chunks += total_chunks
            st.session_state.sample_chunks = sample_chunks
            
            st.success(f"✅ Successfully processed {len(new_files)} files into {total_chunks} chunks!")
            
            # Show sample chunks
            if sample_chunks:
                with st.expander("👀 View Sample Chunks (see how your text was split)"):
                    for i, chunk in enumerate(sample_chunks, 1):
                        st.markdown(f"**Chunk {i}:** ({len(chunk)} characters)")
                        st.text_area(f"chunk_{i}", chunk, height=100, disabled=True, label_visibility="collapsed")
    
    # Show how RAG works
    if not st.session_state.processed_files:
        st.info("👆 **Start by uploading PDF files in the sidebar!**")
        
        # Educational info
        st.markdown("---")
        st.subheader("📖 How RAG Works (Step by Step)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>1️⃣ Document Processing</h4>
            <ul>
                <li>Split text into chunks</li>
                <li>Convert to embeddings (numbers)</li>
                <li>Store in vector database</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>2️⃣ Query & Retrieval</h4>
            <ul>
                <li>User asks a question</li>
                <li>Convert question to embedding</li>
                <li>Find similar chunks in database</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
            <h4>3️⃣ Response</h4>
            <ul>
                <li><b>Without AI:</b> Show raw chunks</li>
                <li><b>With AI:</b> Generate answer from chunks</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Chat interface
        st.markdown("---")
        st.subheader("💬 Ask Questions")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if query := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Search for relevant chunks
            with st.spinner("🔍 Searching vector database..."):
                results = rag.search(query, top_k)
            
            if not results['documents'][0]:
                response = "❌ No relevant information found. Try uploading more documents or rephrasing your question."
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Build context from retrieved chunks
                context_chunks = []
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    context_chunks.append({
                        'text': doc,
                        'source': meta['source'],
                        'page': meta['page']
                    })
                
                # Response based on AI enabled or not
                if use_ai and ai_key:
                    # AI-powered response
                    with st.spinner("🤖 AI is generating answer..."):
                        try:
                            context_str = "\n\n".join([f"[{c['source']}, Page {c['page']}]\n{c['text']}" for c in context_chunks])
                            
                            # Call AI based on provider
                            if ai_provider == "OpenAI (GPT)":
                                from openai import OpenAI
                                client = OpenAI(api_key=ai_key)
                                completion = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "Answer based on the provided context. Be clear and concise."},
                                        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
                                    ],
                                    temperature=creativity
                                )
                                answer = completion.choices[0].message.content
                            
                            elif ai_provider == "Hugging Face":
                                import requests
                                headers = {"Authorization": f"Bearer {ai_key}"}
                                payload = {
                                    "inputs": f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:",
                                    "parameters": {"temperature": creativity, "max_new_tokens": 300}
                                }
                                response = requests.post(
                                    "https://api-inference.huggingface.co/models/google/flan-t5-large",
                                    headers=headers,
                                    json=payload
                                )
                                answer = response.json()[0]['generated_text'] if response.status_code == 200 else "Error calling API"
                            
                            elif ai_provider == "Google (Gemini)":
                                import google.genai as genai
                                client = genai.Client(api_key=ai_key)
                                from google.genai.types import GenerateContentConfig
                                response = client.models.generate_content(
                                    model='gemini-1.5-flash',
                                    contents=f"Context:\n{context_str}\n\nQuestion: {query}",
                                    config=GenerateContentConfig(temperature=creativity)
                                )
                                answer = response.text
                            
                            # Display AI response
                            response_text = f"**🤖 AI Answer:**\n\n{answer}\n\n---\n**📚 Sources:**\n"
                            for i, chunk in enumerate(context_chunks, 1):
                                response_text += f"{i}. {chunk['source']} (Page {chunk['page']})\n"
                            
                            st.chat_message("assistant").markdown(response_text)
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                        except Exception as e:
                            error_msg = f"❌ AI Error: {str(e)}\n\nShowing raw chunks instead..."
                            st.chat_message("assistant").markdown(error_msg)
                            use_ai = False  # Fallback to raw chunks
                
                if not use_ai or not ai_key:
                    # Raw RAG response (no AI)
                    response_text = f"**🔍 Found {len(context_chunks)} relevant chunks:**\n\n"
                    
                    for i, chunk in enumerate(context_chunks, 1):
                        response_text += f"**Chunk {i}** ({chunk['source']}, Page {chunk['page']}):\n"
                        response_text += f"```\n{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}\n```\n\n"
                    
                    response_text += "\n💡 **Tip:** Enable AI Assistant in sidebar to get human-readable answers!"
                    
                    st.chat_message("assistant").markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()