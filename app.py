import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# --- Configuration ---
FILE_PATH = "data/Resturaunt Q&A.pdf" 
VECTOR_DB_DIR = "./chroma_db"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "embedding-001"

# --- Function to Initialize the RAG System (Cached) ---

# st.cache_resource ensures this function runs only once,
# even if the user interacts with the app repeatedly.
@st.cache_resource
def initialize_rag_system():
    """Initializes the RAG chain, loading documents and vector store."""
    st.write("Initializing RAG system... (This runs only on startup)")
    
    # 1. Check API Key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY environment variable not set.")
        return None

    # 2. Load and Split Documents
    try:
        loader = PyPDFLoader(FILE_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        st.success(f"Loaded {len(documents)} pages and split into {len(docs)} chunks.")

    except FileNotFoundError:
        st.error(f"ERROR: The file {FILE_PATH} was not found.")
        st.warning("Please create a 'data' directory and place a 'handbook.pdf' inside it.")
        return None

    # 3. Embedding and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM and Chain Setup
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    
    system_prompt = (
        "You are an expert Q&A assistant for corporate documents. "
        "Use ONLY the following retrieved context to answer the user's question. "
        "Be concise and professional. "
        "If the answer is not found in the context, strictly respond with: "
        "'I cannot find that information in the provided documents.'"
        "\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    return rag_chain

# --- Function to Run the RAG Chain ---

def get_rag_response(rag_chain, user_input):
    """Invokes the RAG chain and returns the clean text answer."""
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input})
    
    # Extract clean text from the response structure
    final_message_content = response["answer"]
    clean_text_answer = final_message_content[0]['text'] if isinstance(final_message_content, list) else final_message_content
    
    return clean_text_answer

# --- Streamlit UI App ---

def main():
    st.set_page_config(page_title="Gemini RAG Chatbot POC", layout="wide")
    st.title("📄 Gemini RAG Chatbot (Local POC)")
    st.markdown("Ask a question based on the content of your `handbook.pdf`.")

    # Initialize the RAG system only once
    rag_chain = initialize_rag_system()

    if rag_chain is None:
        st.stop() # Stop if initialization failed (e.g., missing API key or PDF)

    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about the company handbook..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the RAG chain
        response = get_rag_response(rag_chain, prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()