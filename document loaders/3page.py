# ab kuch website mei boht info toh m nhi pdna chahta
# toh i will use webbaseloader to load the data from website and 
# then use it for my purpose. iske liy chahie url.

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

docs = data.load()

print(len(docs))

# agr multiple url hai toh list mei dalo aur loop chla ke ek ek krke bhejdo.