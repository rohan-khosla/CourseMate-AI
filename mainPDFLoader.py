from langchain_community.document_loaders import PyPDFLoader 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
# chat prompt templlate mei hm roles bhi de skte ai ko.

load_dotenv()

data = PyPDFLoader("document loaders/DeepLearningBook.pdf")
docs = data.load()  

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the text."),
    ("human", "{data}"),
])

model = init_chat_model("google_genai:gemini-3.5-flash-lite", 
                        temperature=0.1, max_output_tokens=100)

prompt = template.format_prompt(data=docs[0].page_content)

# Test the model
response = model.invoke(prompt)

print(response.content[0]["text"])

'''ab agr directly bina kuch split kiy bhej dnge toh dikkat ayegi context
window ki toh chunks mei split krke bhejege.

abhi toh single single page krke bhejre toh koi dikkat ni ayegi.
'''