import os

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import UpstageDocumentParseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from fastapi import BackgroundTasks

# upstage models
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "galaxy-a35"
pdf_path = "Galaxy_A_35.pdf"

load_dotenv()

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    qa = RetrievalQA.from_chain_type(llm=chat_upstage,
                                     chain_type="stuff",
                                     retriever=pinecone_retriever,
                                     return_source_documents=True)

    result = qa(req.message)
    return {"reply": result['result']}


# @app.post("/assistant")
# async def assistant_endpoint(req: AssistantRequest):
#     assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")
#
#     if req.thread_id:
#         # We have an existing thread, append user message
#         await openai.beta.threads.messages.create(
#             thread_id=req.thread_id, role="user", content=req.message
#         )
#         thread_id = req.thread_id
#     else:
#         # Create a new thread with user message
#         thread = await openai.beta.threads.create(
#             messages=[{"role": "user", "content": req.message}]
#         )
#         thread_id = thread.id
#
#     # Run and wait until complete
#     await openai.beta.threads.runs.create_and_poll(
#         thread_id=thread_id, assistant_id=assistant.id
#     )
#
#     # Now retrieve messages for this thread
#     # messages.list returns an async iterator, so let's gather them into a list
#     all_messages = [
#         m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
#     ]
#     print(all_messages)
#
#     # The assistant's reply should be the last message with role=assistant
#     assistant_reply = all_messages[0].content[0].text.value
#
#     return {"reply": assistant_reply, "thread_id": thread_id}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


def embed():
    print("start")
    document_parse_loader = UpstageDocumentParseLoader(
        pdf_path,
        output_format='html',  # 결과물 형태 : HTML
        coordinates=False)  # 이미지 OCR 좌표계 가지고 오지 않기

    docs = document_parse_loader.load()

    # Split the document into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100)

    # Embed the splits

    splits = text_splitter.split_documents(docs)

    PineconeVectorStore.from_documents(
        splits, embedding_upstage, index_name=index_name
    )
    print("end")


@app.post("/embed")
async def load_pdf():
    embed()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
