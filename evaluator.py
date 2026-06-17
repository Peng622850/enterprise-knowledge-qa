import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    relevancy: int      # 相关性 1-5
    faithfulness: int   # 忠实度 1-5
    completeness: int   # 完整性 1-5
    comment: str        # 评语


def evaluate_answer(question: str, answer: str, contexts: list[str]) -> EvalResult:
    """
    用 LLM 对 RAG 系统的回答进行评测
    - 相关性：答案是否切题
    - 忠实度：答案是否忠实于检索内容，没有编造
    - 完整性：答案是否完整回答了问题
    """
    context_str = "\n---\n".join(contexts) if contexts else "无"

    prompt = f"""你是一个RAG系统评测专家，请对以下问答进行评分。

问题：{question}

检索到的知识库内容：
{context_str}

系统回答：{answer}

请从以下三个维度打分（1-5分，5分最高）：
1. 相关性：答案是否直接回答了问题
2. 忠实度：答案是否完全基于知识库内容，没有编造
3. 完整性：答案是否完整，没有遗漏重要信息

请严格按照以下JSON格式输出，不要输出其他内容：
{{
    "relevancy": 分数,
    "faithfulness": 分数,
    "completeness": 分数,
    "comment": "简短评语"
}}"""

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    import json
    raw = response.choices[0].message.content.strip()
    # 清理可能的 markdown 代码块
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    result = EvalResult(
        question=question,
        answer=answer,
        relevancy=data["relevancy"],
        faithfulness=data["faithfulness"],
        completeness=data["completeness"],
        comment=data["comment"],
    )
    logger.info(f"评测结果：相关性={result.relevancy} 忠实度={result.faithfulness} 完整性={result.completeness}")
    return result


def batch_evaluate(test_cases: list[dict]) -> dict:
    """
    批量评测，输入格式：
    [
        {"question": "...", "answer": "...", "contexts": ["...", "..."]}
    ]
    """
    results = []
    for case in test_cases:
        result = evaluate_answer(
            question=case["question"],
            answer=case["answer"],
            contexts=case.get("contexts", []),
        )
        results.append(result)

    avg_relevancy = sum(r.relevancy for r in results) / len(results)
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_completeness = sum(r.completeness for r in results) / len(results)

    return {
        "total": len(results),
        "avg_relevancy": round(avg_relevancy, 2),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_completeness": round(avg_completeness, 2),
        "details": [
            {
                "question": r.question,
                "relevancy": r.relevancy,
                "faithfulness": r.faithfulness,
                "completeness": r.completeness,
                "comment": r.comment,
            }
            for r in results
        ]
    }