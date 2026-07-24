import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Graph RAG QA", page_icon="🧠", layout="centered")
st.title("🧠 Multi-Document QA System")
st.caption("Powered by Graph RAG (Retrieval Augmented Generation)")

# --- SAMPLE DOCUMENTS (Our Database) ---
documents = {
    "AI_Basics.txt": "Artificial Intelligence is the simulation of human intelligence in machines. Machine Learning is a subset of AI that allows systems to learn from data. Deep Learning is a part of Machine Learning that uses neural networks.",
    "NLP_Intro.txt": "Natural Language Processing is a branch of AI. NLP helps computers understand human language. Key parts of NLP are text classification and named entity recognition. Transformers are the backbone of modern NLP.",
    "Graph_Databases.txt": "Graph databases store data as nodes and relationships. Neo4j is a popular graph database. Knowledge graphs connect information like a mind map. Graphs are useful for recommendation systems."
}

# --- LOAD MODELS (Cached so it doesn't reload every time you click a button) ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    return embedder, qa_tokenizer, qa_model

# --- BUILD GRAPH RAG SYSTEM ---
@st.cache_resource
def build_graph_rag():
    embedder, qa_tokenizer, qa_model = load_models()
    
    # 1. Chunking
    chunks = []
    chunk_id = 0
    for doc_name, text in documents.items():
        sentences = text.strip().split(". ")
        for sentence in sentences:
            if len(sentence) > 10:
                chunks.append({"id": chunk_id, "text": sentence, "source": doc_name})
                chunk_id += 1

    # 2. Embeddings
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    # 3. Build Graph
    graph = {i: [] for i in range(len(chunks))}
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            words_i = set(chunks[i]["text"].lower().split())
            words_j = set(chunks[j]["text"].lower().split())
            common_words = words_i.intersection(words_j)
            if len(common_words) >= 2:
                graph[i].append(j)
                graph[j].append(i)
                
    return chunks, embeddings, graph, embedder, qa_tokenizer, qa_model

# --- MAIN APP LOGIC ---
# This message shows while the AI models are downloading
with st.spinner("🧠 Loading AI Models & Building Knowledge Graph... (Takes about 1 minute on first load)"):
    chunks, embeddings, graph, embedder, qa_tokenizer, qa_model = build_graph_rag()

st.success("Ask me anything about AI, NLP, or Graphs.")

# --- USER INTERFACE ---
question = st.text_input("Enter your question:", placeholder="e.g., What is Deep Learning?")

if st.button("Get Answer") and question:
    with st.spinner("Searching Graph & Generating Answer..."):
        # Step A: Vector Search
        q_embedding = embedder.encode([question])[0]
        similarities = np.dot(embeddings, q_embedding)
        best_chunk_id = np.argmax(similarities)
        
        # Step B: Graph Traversal
        context_chunks = [chunks[best_chunk_id]["text"]]
        sources_used = [chunks[best_chunk_id]["source"]]
        
        for neighbor_id in graph[best_chunk_id]:
            context_chunks.append(chunks[neighbor_id]["text"])
            sources_used.append(chunks[neighbor_id]["source"])
        
        final_context = " ".join(context_chunks)
        
        # Step C: Generate Answer
        inputs = qa_tokenizer(question, final_context, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = qa_model(**inputs)
            
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1
        answer = qa_tokenizer.decode(inputs.input_ids[0, answer_start:answer_end])
        
        # Display Results
        st.markdown("### Answer:")
        st.info(answer)
        
        st.markdown("### Sources Retrieved from Graph:")
        for src in set(sources_used):
            st.markdown(f"- `{src}`")

# Footer
st.markdown("---")
st.markdown("Built for College Project | Graph RAG Architecture")
