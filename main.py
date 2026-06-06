import logging

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import agent
from file_parser import parse_file
from rag import add_documents

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="企业知识库问答系统")

app.mount("/static", StaticFiles(directory="static"), name="static")


class QuestionRequest(BaseModel):
    question: str


class AddDocRequest(BaseModel):
    texts: list[str]


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add")
def add(req: AddDocRequest):
    count = add_documents(req.texts)
    return {"message": f"成功写入 {count} 个片段"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), category: str = "通用"):
    file_bytes = await file.read()
    try:
        text = parse_file(file.filename, file_bytes)
    except ValueError as e:
        return {"error": str(e)}

    if not text.strip():
        return {"error": "文件内容为空，请检查文件"}

    count = add_documents(
        texts=[text],
        metadatas=[{"source": file.filename, "category": category}]
    )
    logger.info(f"文件上传成功：{file.filename}，分类：{category}，{count} 个 chunk")
    return {
        "message": f"文件《{file.filename}》上传成功",
        "category": category,
        "chunks": count
    }


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/chat")
def chat(req: QuestionRequest):
    def stream():
        config = {"configurable": {"thread_id": req.session_id}}
        for chunk, metadata in agent.stream(
            {"messages": [HumanMessage(content=req.question)]},
            config=config,
            stream_mode="messages",
        ):
            if (
                hasattr(chunk, "content")
                and chunk.content
                and metadata.get("langgraph_node") == "agent"
                and not getattr(chunk, "tool_calls", None)
            ):
                yield chunk.content

    return StreamingResponse(stream(), media_type="text/plain")