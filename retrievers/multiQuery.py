from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()


llm = init_chat_model("google_genai:gemini-3.5-flash-lite", 
                        temperature=0.1, max_output_tokens=100)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)