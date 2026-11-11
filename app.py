import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its content.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Initialize Gemini
@st.cache_resource
def get_llm():
    return init_chat_model(
        "google_genai:gemini-3.5-flash-lite",
        max_output_tokens=500
    )

embedding_model = get_embedding_model()
llm = get_llm()

# PDF upload section
st.subheader("Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Processing your PDF..."):

            # Load PDF
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))

            docs = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={
                        "source": uploaded_file.name,
                        "page": i + 1
                    }
                )
                for i, page in enumerate(reader.pages)
            ]

            # Remove empty pages
            docs = [
                doc for doc in docs
                if doc.page_content.strip()
            ]

            if not docs:
                st.error("No readable text found in this PDF.")
                st.stop()

            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # Create vector database
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []

            st.success(
                f"PDF processed successfully! "
                f"{len(chunks)} chunks created."
            )

# Chat section
if st.session_state.vectorstore is not None:

    st.divider()
    st.subheader("Ask questions about your PDF")

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

            Answer using ONLY the provided context.

            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}
            """
        )
    ])

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    query = st.chat_input("Ask something about your PDF...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Finding the answer..."):

                # Retrieve relevant chunks
                retrieved_docs = retriever.invoke(query)

                context = "\n\n".join(
                    [
                        f"Page {doc.metadata.get('page', 'Unknown')}:\n"
                        f"{doc.page_content}"
                        for doc in retrieved_docs
                    ]
                )

                # Generate final prompt
                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                # Generate answer
                response = llm.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

                # Show source pages
                pages = sorted(set(
                    doc.metadata.get("page")
                    for doc in retrieved_docs
                    if doc.metadata.get("page") is not None
                ))

                st.caption(f"Retrieved from PDF pages: {pages}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
