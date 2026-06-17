# eval_compare.py
# 把这个文件放到 D:\FastAPIProject2\ 目录下，然后运行：
# python eval_compare.py
#
# 作用：对比三种检索方式的效果
#   方式1：纯向量检索
#   方式2：向量 + BM25 + RRF 混合检索（不含Rerank/MultiQuery）
#   方式3：完整链路（混合 + MultiQuery + Rerank）
# 每种方式都用 LLM-as-Judge 打分，最后输出对比表格

import json
import os
import time

import jieba
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from evaluator import evaluate_answer
from rag import (
    get_vectorstore,
    get_bm25,
    rrf_fusion,
    search,          # 完整链路（方式3）
    llm_client,
    _tokenize,
)

load_dotenv()

# ──────────────────────────────────────────────
# 测试问题（基于你上传的 Python 技术文档来写）
# 上传文档后，把这里的问题替换成文档里真实涉及的内容
# ──────────────────────────────────────────────
TEST_QUESTIONS = [
    "Python中list和tuple的区别是什么？",
    "如何使用装饰器？",
    "什么是GIL，它对多线程有什么影响？",
    "Python的垃圾回收机制是怎样的？",
    "生成器和迭代器的区别是什么？",
    "如何处理Python中的异常？",
    "什么是上下文管理器，如何自定义？",
    "深拷贝和浅拷贝的区别？",
    "Python中的*args和**kwargs有什么用？",
    "如何使用asyncio实现异步编程？",
]

TOP_K = 3  # 每次检索返回的chunk数量


# ──────────────────────────────────────────────
# 方式1：纯向量检索（不用BM25，不用Rerank，不用MultiQuery）
# ──────────────────────────────────────────────
def search_vector_only(query: str, top_k: int = TOP_K) -> list[str]:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [doc.page_content for doc, score in results if score < 20]


# ──────────────────────────────────────────────
# 方式2：混合检索（BM25 + 向量 + RRF），不含MultiQuery和Rerank
# ──────────────────────────────────────────────
def search_hybrid_only(query: str, top_k: int = TOP_K) -> list[str]:
    vectorstore = get_vectorstore()
    vec_results_raw = vectorstore.similarity_search_with_score(query, k=top_k)
    vec_results = [doc.page_content for doc, score in vec_results_raw if score < 20]

    bm25, corpus = get_bm25()
    if bm25 and corpus:
        tokenized_query = _tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k * 2]
        bm25_results = [corpus[i] for i in top_indices if bm25_scores[i] > 0]
    else:
        bm25_results = []

    fused = rrf_fusion(vec_results, bm25_results)
    return fused[:top_k]


# ──────────────────────────────────────────────
# 用LLM生成回答（三种方式都用同一个生成逻辑，只是检索结果不同）
# ──────────────────────────────────────────────
def generate_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "知识库中未找到相关内容。"

    context_str = "\n---\n".join(contexts)
    prompt = f"""请根据以下知识库内容回答用户问题。只使用知识库中的信息，不要编造。

知识库内容：
{context_str}

用户问题：{question}

请给出准确、完整的回答："""

    response = llm_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# 主评测流程
# ──────────────────────────────────────────────
def run_comparison():
    print("=" * 60)
    print("RAG 检索方式对比评测")
    print("=" * 60)
    print(f"测试问题数量：{len(TEST_QUESTIONS)}")
    print(f"每次检索 Top-K：{TOP_K}")
    print()

    results = {
        "vector_only": [],
        "hybrid": [],
        "full_pipeline": [],
    }

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] 问题：{question}")
        print("-" * 40)

        # ── 方式1：纯向量 ──
        print("  ▶ 方式1：纯向量检索...")
        ctx1 = search_vector_only(question)
        ans1 = generate_answer(question, ctx1)
        eval1 = evaluate_answer(question, ans1, ctx1)
        results["vector_only"].append({
            "question": question,
            "contexts_count": len(ctx1),
            "relevancy": eval1.relevancy,
            "faithfulness": eval1.faithfulness,
            "completeness": eval1.completeness,
            "comment": eval1.comment,
        })
        print(f"     相关性={eval1.relevancy} 忠实度={eval1.faithfulness} 完整性={eval1.completeness}")
        time.sleep(1)  # 避免API限速

        # ── 方式2：混合检索 ──
        print("  ▶ 方式2：混合检索(BM25+向量+RRF)...")
        ctx2 = search_hybrid_only(question)
        ans2 = generate_answer(question, ctx2)
        eval2 = evaluate_answer(question, ans2, ctx2)
        results["hybrid"].append({
            "question": question,
            "contexts_count": len(ctx2),
            "relevancy": eval2.relevancy,
            "faithfulness": eval2.faithfulness,
            "completeness": eval2.completeness,
            "comment": eval2.comment,
        })
        print(f"     相关性={eval2.relevancy} 忠实度={eval2.faithfulness} 完整性={eval2.completeness}")
        time.sleep(1)

        # ── 方式3：完整链路 ──
        print("  ▶ 方式3：完整链路(+MultiQuery+Rerank)...")
        ctx3 = search(question, top_k=TOP_K)  # 直接用rag.py里的完整search函数
        ans3 = generate_answer(question, ctx3)
        eval3 = evaluate_answer(question, ans3, ctx3)
        results["full_pipeline"].append({
            "question": question,
            "contexts_count": len(ctx3),
            "relevancy": eval3.relevancy,
            "faithfulness": eval3.faithfulness,
            "completeness": eval3.completeness,
            "comment": eval3.comment,
        })
        print(f"     相关性={eval3.relevancy} 忠实度={eval3.faithfulness} 完整性={eval3.completeness}")
        time.sleep(1)

    # ──────────────────────────────────────────────
    # 汇总输出
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)

    summary = {}
    for method, data in results.items():
        n = len(data)
        avg_r = sum(d["relevancy"] for d in data) / n
        avg_f = sum(d["faithfulness"] for d in data) / n
        avg_c = sum(d["completeness"] for d in data) / n
        avg_total = (avg_r + avg_f + avg_c) / 3
        summary[method] = {
            "avg_relevancy": round(avg_r, 2),
            "avg_faithfulness": round(avg_f, 2),
            "avg_completeness": round(avg_c, 2),
            "avg_total": round(avg_total, 2),
        }

    method_names = {
        "vector_only": "方式1：纯向量检索",
        "hybrid": "方式2：混合检索(BM25+RRF)",
        "full_pipeline": "方式3：完整链路(+MultiQuery+Rerank)",
    }

    print(f"\n{'检索方式':<35} {'相关性':^6} {'忠实度':^6} {'完整性':^6} {'综合':^6}")
    print("-" * 65)
    for method, s in summary.items():
        print(
            f"{method_names[method]:<35} "
            f"{s['avg_relevancy']:^6} "
            f"{s['avg_faithfulness']:^6} "
            f"{s['avg_completeness']:^6} "
            f"{s['avg_total']:^6}"
        )

    # 提升幅度
    v = summary["vector_only"]
    f = summary["full_pipeline"]
    print(f"\n完整链路 vs 纯向量，综合得分提升：{round(f['avg_total'] - v['avg_total'], 2)} 分")
    print(f"完整链路 vs 纯向量，相关性提升：{round(f['avg_relevancy'] - v['avg_relevancy'], 2)} 分")
    print(f"完整链路 vs 纯向量，忠实度提升：{round(f['avg_faithfulness'] - v['avg_faithfulness'], 2)} 分")
    print(f"完整链路 vs 纯向量，完整性提升：{round(f['avg_completeness'] - v['avg_completeness'], 2)} 分")

    # 保存详细结果到JSON
    output = {
        "summary": summary,
        "details": results,
    }
    with open("eval_results.json", "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到 eval_results.json")
    print("\n✅ 评测完成！把上面的汇总表格截图，就是简历里的量化数据。")


if __name__ == "__main__":
    # 运行前确认知识库里有文档
    from rag import _bm25_cache
    if not _bm25_cache:
        print("⚠️  警告：知识库为空！请先上传Python技术文档，再运行评测。")
        print("   上传方法：启动主项目(uvicorn main:app)，通过接口上传文档")
        print("   或者运行下面的快速导入脚本先导入几篇文档")
    else:
        print(f"✅ 知识库已有 {len(_bm25_cache)} 个chunk，开始评测...")
        run_comparison()
