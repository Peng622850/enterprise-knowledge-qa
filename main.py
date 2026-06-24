
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

from eval_storage import save_eval_run, list_eval_runs, get_eval_run
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from auth import get_current_tenant, create_token
from fastapi import Depends

load_dotenv()

from logger_setup import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="企业知识库问答系统")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    status = {"status": "ok", "dependencies": {}}

    # 检查 Redis
    try:
        from rag import redis_client
        redis_client.ping()
        status["dependencies"]["redis"] = "ok"
    except Exception as e:
        status["dependencies"]["redis"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # 检查 ChromaDB
    try:
        from rag import get_vectorstore
        vs = get_vectorstore()
        vs.get(limit=1)
        status["dependencies"]["chromadb"] = "ok"
    except Exception as e:
        status["dependencies"]["chromadb"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # 检查 LLM API（轻量 ping，不实际生成）
    try:
        from rag import llm_client
        llm_client.models.list()
        status["dependencies"]["llm_api"] = "ok"
    except Exception as e:
        status["dependencies"]["llm_api"] = f"error: {str(e)}"
        status["status"] = "degraded"

    return status


@app.post("/add")
def add(req: AddDocRequest):
    count = add_documents(req.texts)
    return {"message": f"成功写入 {count} 个片段"}


from fastapi import Form

@app.post("/upload")
async def upload(file: UploadFile = File(...), category: str = Form(default="通用"),tenant: dict = Depends(get_current_tenant)):
    file_bytes = await file.read()
    try:
        text, block_metadatas = parse_file(file.filename, file_bytes)
    except ValueError as e:
        return {"error": str(e)}

    if not text.strip():
        return {"error": "文件内容为空，请检查文件"}

        # 给每个块补上 category
    for m in block_metadatas:
        m["category"] = category

    count = add_documents(
        texts=[text],
        metadatas=[{"source": file.filename, "category": category}],
        tenant_id=tenant["tenant_id"]
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
@limiter.limit("20/minute")
def chat(req: QuestionRequest, request: Request, tenant: dict = Depends(get_current_tenant)):
    def stream():
        from agent import agent
        from langchain_core.messages import HumanMessage

        config = {"configurable": {"thread_id": f"{tenant['tenant_id']}:{req.session_id}"}}

        try:
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=req.question)]},
                config=config,
                stream_mode="messages",
            ):
                # chunk 是 (message, metadata) 元组
                msg, metadata = chunk
                if (
                    metadata.get("langgraph_node") == "agent"
                    and hasattr(msg, "content")
                    and msg.content
                ):
                    yield msg.content
        except Exception as e:
            logger.error(f"Agent 执行失败：{e}")
            yield f"系统错误，请稍后重试。"

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
    run_id = save_eval_run(
        params={"top_k": 3, "chunk_size": 500, "rerank": True, "model": "DeepSeek-V3"},
        summary={
            "avg_relevancy": result["avg_relevancy"],
            "avg_faithfulness": result["avg_faithfulness"],
            "avg_completeness": result["avg_completeness"],
            "avg_total": round(
                (result["avg_relevancy"] + result["avg_faithfulness"] + result["avg_completeness"]) / 3, 2
            ),
        },
        details=result["details"],
    )
    result["run_id"] = run_id
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


@app.get("/eval/history")
def eval_history():
    return list_eval_runs()

@app.get("/eval/run/{run_id}")
def eval_run_detail(run_id: int):
    run = get_eval_run(run_id)
    if not run:
        return {"error": "未找到该 run"}
    return run

@app.get("/eval/compare")
def eval_compare(run_a: int, run_b: int):
    a = get_eval_run(run_a)
    b = get_eval_run(run_b)
    if not a or not b:
        return {"error": "run_id 不存在"}
    return {
        "run_a": {"id": a["id"], "created_at": a["created_at"], "params": a["params"], "avg_total": a["avg_total"]},
        "run_b": {"id": b["id"], "created_at": b["created_at"], "params": b["params"], "avg_total": b["avg_total"]},
        "diff": {
            "relevancy": round(b["avg_relevancy"] - a["avg_relevancy"], 2),
            "faithfulness": round(b["avg_faithfulness"] - a["avg_faithfulness"], 2),
            "completeness": round(b["avg_completeness"] - a["avg_completeness"], 2),
            "total": round(b["avg_total"] - a["avg_total"], 2),
        }
    }
@app.post("/token")
def get_token(tenant_id: str, user_id: str):
    from auth import create_token
    return {"token": create_token(tenant_id, user_id)}