from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]


vectorstore = Chroma.from_documents(
    documents = docs,
    embedding= embedding_model,
    persist_directory= "chroma-db"
)
# hm data locally store jo chroma db folder bn ke usi mei store hogya.

result = vectorstore.similarity_search("what is used for data analysis?",k=2)
# vector store are not responsible for answering your ques these are 
# for retrieving info. 

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()
# for retriving info

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)