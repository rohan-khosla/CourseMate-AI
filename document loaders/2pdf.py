from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter


data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()
print(len(docs))  # print the number of documents loaded = 15(no. of pages in gru)

'''ab hmare pass jo list hogi usmei 15 docs honge, har doc ka apna metadata 
hoga aur pageContent hoga, hum indexing karke access kr skte hai.
'''

# to choose specific page let last page
# print(docs[14]) 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

# print(chunks[0].page_content)
print(len(chunks))