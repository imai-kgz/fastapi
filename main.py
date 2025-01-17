import json
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI

# ✅ Load the JSON file
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# ✅ Convert QA pairs into Documents
docs = [Document(page_content=pair["instruction"], metadata={"answer": pair["output"]}) for pair in qa_pairs]

# ✅ Split the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# ✅ Use a multilingual embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ✅ Create and save FAISS vector store
vector_store = FAISS.from_documents(split_docs, embedding_model)
vector_store.save_local("faiss_index")

# ✅ Load the vector store
vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Retrieve answer from vector store with exact matching and similarity threshold
def retrieve_answer(question):
    # 1️⃣ Exact matching check
    for pair in qa_pairs:
        if pair["instruction"].strip().lower() == question.strip().lower():
            return pair["output"]

    # 2️⃣ Similarity matching with threshold
    results = vector_store.similarity_search_with_score(question, k=1)
    if results:
        top_result, score = results[0]
        # Set a threshold to ensure the answer is relevant
        # print(score)
        if score < 1.25:
            return top_result.metadata.get("answer", "Жооп табылган жок.")
    
    # 3️⃣ Return a default response if no relevant match is found
    return "Сурооңуз боюнча маалымат табылган жок. Сурооңузду тактап, кайта сураңыз. "



# ✅ Clean up the response
def clean_response(response):
    return response.strip()

# ✅ Set up FastAPI
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def query_api(request: QuestionRequest):
    question = request.question
    answer = retrieve_answer(question)
    return {"question": question, "answer": answer}

