import logging

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import agent
from file_parser import parse_file
from rag import add_documents,delete_document, update_document

from evaluator import batch_evaluate

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
        from agent import save_message, load_history

        # 加载历史记录
        history = load_history(req.session_id)
        messages = history + [HumanMessage(content=req.question)]

        config = {"configurable": {"thread_id": req.session_id}}
        full_answer = ""

        for chunk, metadata in agent.stream(
                {"messages": messages},
                config=config,
                stream_mode="messages",
        ):
            if (
                    hasattr(chunk, "content")
                    and chunk.content
                    and metadata.get("langgraph_node") == "agent"
                    and not getattr(chunk, "tool_calls", None)
            ):
                full_answer += chunk.content
                yield chunk.content

        # 流式结束后保存对话历史
        if full_answer:
            save_message(req.session_id, "user", req.question)
            save_message(req.session_id, "assistant", full_answer)

    return StreamingResponse(stream(), media_type="text/plain")


class EvalCase(BaseModel):
    question: str
    answer: str
    contexts: list[str] = []

class EvalRequest(BaseModel):
    cases: list[EvalCase]

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    test_cases = [
        {"question": c.question, "answer": c.answer, "contexts": c.contexts}
        for c in req.cases
    ]
    result = batch_evaluate(test_cases)
    return result

class DeleteRequest(BaseModel):
    source: str

class UpdateRequest(BaseModel):
    source: str
    new_text: str
    category: str = "通用"

@app.post("/delete")
def delete(req: DeleteRequest):
    count = delete_document(req.source)
    if count == 0:
        return {"message": f"未找到来源为《{req.source}》的文档"}
    return {"message": f"已删除《{req.source}》，共删除 {count} 个片段"}

@app.post("/update")
def update(req: UpdateRequest):
    count = update_document(
        source=req.source,
        new_text=req.new_text,
        metadata={"source": req.source, "category": req.category}
    )
    return {"message": f"已更新《{req.source}》，重新生成 {count} 个片段"}