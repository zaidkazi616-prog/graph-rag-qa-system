 Multi-Document QA System using Graph RAG
A smart Question-Answering system that reads multiple documents, builds a Knowledge Graph, and uses Graph RAG architecture to provide highly accurate answers.

🚀 Live Demo
(Will add the hosted link here after Step 4)

📄 How It Works
This project uses a Graph RAG (Retrieval Augmented Generation) pipeline:

Document Ingestion: Loads multiple text documents.
Chunking: Splits documents into smaller, manageable sentences.
Knowledge Graph Creation: Connects chunks that share common entities (like a mind map of information).
Vector Search: Converts questions into numbers (embeddings) to find the most relevant chunk.
Graph Traversal: Uses the found chunk to traverse the graph and grab connected context.
Answer Generation: Feeds the combined context to a DistilBERT AI model to generate the final answer.
🛠️ Tech Stack
Language: Python
NLP Models: HuggingFace Transformers (DistilBERT, Sentence-Transformers)
UI / Frontend: Streamlit
Hosting: Streamlit Community Cloud
