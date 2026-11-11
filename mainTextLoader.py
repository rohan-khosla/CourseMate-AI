from langchain_community.document_loaders import TextLoader 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
# chat prompt templlate mei hm roles bhi de skte ai ko.

load_dotenv()

data = TextLoader("document loaders/notes.txt")
docs = data.load()  

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