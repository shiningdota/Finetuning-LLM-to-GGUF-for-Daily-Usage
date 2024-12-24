import streamlit as st
from openai import OpenAI
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Load the embedding model
device = "cpu"
embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True).to(device)

# Function to read PDF files
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read TXT files
def read_txt(file):
    return file.read().decode("utf-8")

# Function to read DOCX files
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to generate embeddings from text
def generate_embeddings(text):
    prefixed_text = f"search_document: {text}"
    return embedding_model.encode(prefixed_text)

# Function to process the uploaded document and get response from LLM
def process_document(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        text = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        text = read_txt(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        text = read_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None
    

    # Simulate a retrieval process (for demonstration, using the same document)
    retrieved_text = text  # Here you would retrieve relevant context based on embeddings

    return retrieved_text

# Function to query LLM
def query_llm(context, question):
    system_message = "You are an assistant designed to answer questions based on provided documents. Use the following information to respond accurately to the user's question."
    prompt = f"{system_message}\n\nDocument:\n{context}\n\nQuestion: {question}\nResponse:"
    response = client.completions.create(
        model="project-akhir-llama-3.2-3b@q8",
        prompt=prompt,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=1.1
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.set_page_config(page_title="RAG System", layout="wide")
st.title("RAG System with Document Upload")

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

# Main content
if uploaded_file is not None:
    st.success("Document uploaded successfully!")
    context = process_document(uploaded_file)
    
    if context:
        st.text_area("Document Content", context, height=200, disabled=True)

        # User question
        question = st.text_input("Ask a question about the document:")
        if question:
            st.write("Processing your question...")
            response = query_llm(context, question)
            st.subheader("Response:")
            st.write(response)
else:
    st.info("Upload a document to get started.")

