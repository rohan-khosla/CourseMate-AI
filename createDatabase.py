# load pdf
# split into chunks
# create the embeddings
# store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# data = PyPDFLoader("document loaders/DeepLearningBook.pdf")

data = TextLoader("vectorStore/contentForDatabase.txt")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)