--------------------------------EXPLANATION--------------------------------

# CourseMate AI – AI-Powered Study Assistant

CourseMate AI is an AI-powered study assistant that allows
students to interact with their study materials through
natural language using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF study materials.
- Extract and process document text.
- Split documents into overlapping chunks.
- Generate embeddings using Hugging Face.
- Store document embeddings in ChromaDB.
- Retrieve relevant and diverse chunks using MMR.
- Generate context-aware answers using Google Gemini.
- Interactive chat interface built with Streamlit.

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers
- ChromaDB
- Retrieval-Augmented Generation (RAG)

## Architecture

PDF Upload
    ↓
Document Loading
    ↓
Text Splitting
    ↓
Embedding Generation
    ↓
ChromaDB
    ↓
User Query
    ↓
MMR Retrieval
    ↓
Google Gemini
    ↓
Context-Aware Answer

## Installation

1. Clone the repository:

   git clone https://github.com/YOUR_USERNAME/CourseMate-AI.git

2. Navigate to the project:

   cd CourseMate-AI

3. Create a virtual environment:

   python -m venv .venv

4. Activate the environment:

   .venv\Scripts\activate

5. Install dependencies:

   pip install -r requirements.txt

6. Create a .env file and add:

   GOOGLE_API_KEY=your_api_key

7. Run the application:

   streamlit run app.py

