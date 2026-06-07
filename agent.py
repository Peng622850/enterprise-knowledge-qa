import logging
import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

from rag import search, add_documents

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "./chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def load_history(session_id: str, max_turns: int = 10) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT role, content FROM chat_history
           WHERE session_id = ?
           ORDER BY timestamp DESC LIMIT ?""",
        (session_id, max_turns * 2)
    ).fetchall()
    conn.close()
    rows = list(reversed(rows))
    messages = []
    for role, content in rows:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages

init_db()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search_knowledge_base(query: str) -> str:
    """
    在企业知识库中搜索相关信息。
    当用户询问公司内部政策、规章制度、上传文档中的内容时使用此工具。
    """
    results = search(query)
    if not results:
        return "知识库中未找到相关信息。"
    return "\n---\n".join(results)


@tool
def search_web(query: str) -> str:
    """
    联网搜索最新信息。
    当知识库中没有相关内容，或用户询问时事新闻、最新动态、通用知识时使用此工具。
    """
    try:
        response = tavily.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "联网搜索未找到相关信息。"
        texts = []
        for r in results:
            texts.append(f"来源：{r['url']}\n内容：{r['content']}")
        return "\n---\n".join(texts)
    except Exception as e:
        logger.error(f"联网搜索失败：{e}")
        return "联网搜索暂时不可用。"


@tool
def add_to_knowledge_base(content: str, category: str = "通用") -> str:
    """
    将内容写入企业知识库。
    当用户明确要求保存、记录、存入知识库某段内容时使用此工具。
    category 可选值：通用、人事政策、法律法规、财务制度、技术文档
    """
    try:
        count = add_documents(
            texts=[content],
            metadatas=[{"source": "用户输入", "category": category}]
        )
        return f"已成功将内容存入知识库（分类：{category}），共生成 {count} 个片段。"
    except Exception as e:
        return f"写入失败：{str(e)}"


system_prompt = """你是企业知识库助手，同时具备联网搜索和知识库写入能力。

回答规则：
1. 如果问题涉及用户自己的个人信息（姓名、部门等），直接从对话历史中回答，不要查知识库
2. 其他问题先使用 search_knowledge_base 查询知识库
3. 知识库有答案 → 直接回答，注明内容来自知识库
4. 知识库没有答案 → 使用 search_web 联网搜索
5. 用户明确要求保存内容 → 使用 add_to_knowledge_base 写入知识库
6. 两者都没有 → 如实告知用户
7. 不要编造内容，回答简洁准确，使用中文"""

memory = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base, search_web, add_to_knowledge_base],
    prompt=system_prompt,
    checkpointer=memory,
)


def run_agent(question: str, session_id: str = "default") -> str:
    logger.info(f"用户提问：{question}")

    history = load_history(session_id)
    messages = history + [HumanMessage(content=question)]

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": messages},
        config=config,
    )
    answer = result["messages"][-1].content

    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    logger.info(f"Agent 回答：{answer[:50]}...")
    return answer