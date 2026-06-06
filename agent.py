import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

from rag import search

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


system_prompt = """你是企业知识库助手，同时具备联网搜索能力。

回答规则：
1. 先使用 search_knowledge_base 查询知识库
2. 知识库有答案 → 直接回答，注明内容来自知识库
3. 知识库没有答案 → 使用 search_web 联网搜索
4. 两者都没有 → 如实告知用户
5. 不要编造内容，回答简洁准确，使用中文"""

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base, search_web],
    prompt=system_prompt,
    checkpointer=memory,
)

def run_agent(question: str, session_id: str = "default") -> str:
    logger.info(f"用户提问：{question}")
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    answer = result["messages"][-1].content
    logger.info(f"Agent 回答：{answer[:50]}...")
    return answer