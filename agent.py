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

from rag import search, add_documents, llm_client

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
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.3,
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search_knowledge_base(query: str) -> str:
    """
    在企业知识库中搜索相关信息。
    当用户询问公司内部政策、规章制度、上传文档中的内容时使用此工具。
    """
    # 内部自动判断分类，不依赖Agent决策
    classify_prompt = f"""判断以下问题最匹配哪个分类，只输出分类名称，不要解释。

可选分类：人事政策、法律法规、财务制度、技术文档、通用

问题：{query}"""

    classify_resp = llm_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": classify_prompt}],
        temperature=0,
    )
    category = classify_resp.choices[0].message.content.strip()
    logger.info(f"自动分类结果：{category}")

    results = search(query, category=category)

    # 如果分类检索没结果，fallback到全库搜索
    if not results:
        logger.info(f"分类检索无结果，fallback到全库搜索")
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


system_prompt = """你是企业知识库助手。

【强制规则】：每次回答前，无论什么问题，必须先调用 search_knowledge_base 工具查询知识库，这是不可跳过的步骤。

回答流程：
1. 调用 search_knowledge_base 查询知识库（必须执行）
2. 知识库有答案 → 直接回答
3. 知识库没有答案 → 调用 search_web 联网搜索
4. 用户要求保存内容 → 调用 add_to_knowledge_base
5. 不要编造内容，使用中文
6. 直接输出最终回答，不要描述工具调用过程"""

memory = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base, search_web, add_to_knowledge_base],
    prompt=system_prompt,
    checkpointer=memory,
)


def run_agent(question: str, session_id: str = "default") -> str:
    logger.info(f"用户提问：{question}")

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},   # 只传当前问题
        config=config,
    )
    answer = result["messages"][-1].content

    # SQLite 仅做持久化存档，不再参与 LangGraph 输入
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    logger.info(f"Agent 回答：{answer[:50]}...")
    return answer