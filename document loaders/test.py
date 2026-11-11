from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 100,
    chunk_overlap=1
)
'''agr separator htade toh default \n\n hi chunk splitter condition hai aur 
isme #chunks bne kitne = 1 kyuuki hmare notes.txt mei koi bhi \n\n nhi hai. 
hmne single \n condition define kri toh is base pr hoga aur #chunks bdh jayege
'''

data = TextLoader("document loaders/notes.txt") # we need to provide path

# print(data)

docs = data.load()  # load the documents from the file

chunks = splitter.split_documents(docs)

'''ab ye jo docs hai isme 2 contents hoge in form of list
metadata, pageContent of the document, we can access them using indexing

print(docs)  # print the content of the first document
print(docs[0].metadata)  # print the metadata of the first document
print(docs[0].page_content)  # print the content of the first document
'''

# docs isliy bnaye jate kyuki pdf se bhi deal krna hota hai.
print(len(docs))  # print the number of documents loaded