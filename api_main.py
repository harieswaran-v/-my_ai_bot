import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# 1. LOAD API KEY
load_dotenv()
api_key = "AIzaSyDPFpGRi2pXyXPaLabBCH8mAK90BCogaNo" 

#Safety Check: Stop if key is missing
if not api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found in .env file!")
    exit()

app = FastAPI(title="Maxbyte AI API")

# 2. SETUP RAG (This runs once when the server starts)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = "./chroma_db"

# Load document only if database doesn't exist
if not os.path.exists(persist_dir):
    print("Building new vector database from PDF...")
    loader = PyPDFLoader(r"D:\FULL STACK\my_ai_bot\my_documents\user_guide.pdf")
    
    raw_docs = loader.load()
    print("loader:",len(raw_docs))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(raw_docs)
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
else:
    print("Loading existing vector database...")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# FIXED LINE: Pass the api_key explicitly


# 3. DEFINE DATA MODELS
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
async def health():
    return {"status": "ok"}

# 4. THE API ENDPOINT
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    try:
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

        qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})
)
        response = qa_chain.invoke({"query": request.query})
        return ChatResponse(answer=response["result"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))