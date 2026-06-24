import json
import os

import jieba
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from retry_utils import llm_retry

import redis
import hashlib

load_dotenv()

from logger_setup import get_logger
logger = get_logger(__name__)

CHROMA_PATH = "./chroma_db"
BM25_PATH = "./bm25_data.json"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
reranker = CrossEncoder("BAAI/bge-reranker-base")

llm_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CACHE_TTL = 3600  # 缓存1小时

# ── BM25 内存缓存 ──────────────────────────────────────────
_bm25_cache: list[dict] = []   # 全量数据，启动时加载一次

def _load_cache_from_disk():
    """启动时调用一次，把磁盘数据读进内存。"""
    global _bm25_cache
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            _bm25_cache = json.load(f)
    else:
        _bm25_cache = []
    logger.info(f"BM25 缓存加载完毕，共 {len(_bm25_cache)} 条")

_load_cache_from_disk()   # 模块导入时执行
# ──────────────────────────────────────────────────────────


def get_vectorstore(tenant_id: str = "default") -> Chroma:
    collection_name = f"tenant_{tenant_id}"
    return Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )


def load_bm25_data() -> list[dict]:
    """直接返回内存缓存，不再读磁盘。"""
    return _bm25_cache


def save_bm25_data(chunks: list[str], metadata: dict = None):
    """写入时去重，同时热更新内存缓存。"""
    global _bm25_cache

    # 用 (text, source) 做去重键
    existing_keys = {
        (d["text"], d.get("source", ""))
        for d in _bm25_cache
    }

    new_items = []
    for chunk in chunks:
        key = (chunk, metadata.get("source", "未知") if metadata else "未知")
        if key not in existing_keys:
            new_items.append({
                "text": chunk,
                "category": metadata.get("category", "通用") if metadata else "通用",
                "source": metadata.get("source", "未知") if metadata else "未知",
            })
            existing_keys.add(key)

    if not new_items:
        logger.info("BM25：所有 chunk 均已存在，跳过写入")
        return

    _bm25_cache.extend(new_items)          # 热更新内存
    with open(BM25_PATH, "w", encoding="utf-8") as f:
        json.dump(_bm25_cache, f, ensure_ascii=False)
    logger.info(f"BM25：新增 {len(new_items)} 条，总计 {len(_bm25_cache)} 条")


def _tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，过滤空串。"""
    return [t for t in jieba.cut(text) if t.strip()]


def get_bm25(category: str = None) -> tuple[BM25Okapi | None, list[str]]:
    data = _bm25_cache
    if not data:
        return None, []

    if category:
        filtered = [d for d in data if d.get("category") == category]
        corpus = [d["text"] for d in filtered] if filtered else [d["text"] for d in data]
    else:
        corpus = [d["text"] for d in data]

    if not corpus:
        return None, []

    tokenized = [_tokenize(text) for text in corpus]   # jieba 分词
    return BM25Okapi(tokenized), corpus


def add_documents(texts: list[str], metadatas: list[dict] = None, tenant_id: str = "default") -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.create_documents(texts, metadatas=metadatas)

    if metadatas:
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = metadatas[0]

    chunk_texts = [c.page_content for c in chunks]
    vectorstore = get_vectorstore(tenant_id)
    vectorstore.add_documents(chunks)
    save_bm25_data(chunk_texts, metadatas[0] if metadatas else {})
    logger.info(f"写入 {len(chunks)} 个 chunk，分类：{metadatas[0].get('category') if metadatas else '未知'}")
    return len(chunks)


@llm_retry
def _call_llm_generate_queries(prompt: str) -> str:
    response = llm_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def generate_queries(question: str) -> list[str]:
    prompt = f"""请将以下问题改写成3个不同的表达方式，用于检索知识库。
要求：每行一个问题，只输出问题本身，不要编号，不要解释。

原问题：{question}"""
    try:
        raw = _call_llm_generate_queries(prompt)
        queries = [q.strip() for q in raw.split("\n") if q.strip()]
        all_queries = [question] + queries[:3]
        logger.info(f"MultiQuery 改写：{all_queries}")
        return all_queries
    except Exception as e:
        logger.error(f"MultiQuery 失败，降级为单查询：{e}")
        return [question]  # fallback：降级为原始问题


def rrf_fusion(vec_results: list[str], bm25_results: list[str], k: int = 60) -> list[str]:
    scores = {}
    for rank, text in enumerate(vec_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
    for rank, text in enumerate(bm25_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_results]


def search(query: str, top_k: int = 3, category: str = None, tenant_id: str = "default") -> list[str]:
    import numpy as np
    logger.info(f"search函数开始执行，query={query[:20]}")

    # 把当前问题向量化
    query_vec = embeddings.embed_query(query)
    query_vec = np.array(query_vec)

    # 去Redis查所有缓存的key
    SIMILARITY_THRESHOLD = 0.85
    cached_keys = []
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, match="rag:vec:*", count=100)
        cached_keys.extend(keys)
        if cursor == 0:
            break
    logger.info(f"Redis中共有{len(cached_keys)}条语义缓存")

    for key in cached_keys:
        cached_data = redis_client.get(key)
        if not cached_data:
            continue
        data = json.loads(cached_data)
        cached_vec = np.array(data["vector"])

        # 计算余弦相似度
        similarity = np.dot(query_vec, cached_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-9
        )

        if similarity >= SIMILARITY_THRESHOLD:
            logger.info(f"语义缓存命中（相似度{similarity:.3f}）：{query[:20]}")
            return data["results"]
        logger.info(f"缓存未命中，相似度{similarity:.3f}")

    # 没命中，走完整检索链路
    queries = generate_queries(query)
    vectorstore = get_vectorstore(tenant_id)

    all_vec_results = []
    for q in queries:
        if category:
            results = vectorstore.similarity_search_with_score(
                q, k=top_k,
                filter={"category": category}
            )
        else:
            results = vectorstore.similarity_search_with_score(q, k=top_k)
        all_vec_results.extend(
            [doc.page_content for doc, score in results if score < 20]
        )

    seen = set()
    vec_results = []
    for r in all_vec_results:
        if r not in seen:
            seen.add(r)
            vec_results.append(r)

    bm25, corpus = get_bm25(category=category)
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

    candidates = rrf_fusion(vec_results, bm25_results)[:top_k * 3]
    logger.info(f"candidates数量：{len(candidates)}，vec:{len(vec_results)}，bm25:{len(bm25_results)}")
    if not candidates:
        return []

    pairs = [[query, chunk] for chunk in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    results = [text for text, _ in ranked[:top_k]]

    # 存入语义缓存：key用MD5，value存向量+结果
    cache_key = f"rag:vec:{hashlib.md5((query + str(category)).encode()).hexdigest()}"
    cache_data = {
        "vector": query_vec.tolist(),
        "results": results,
        "query": query,
    }
    redis_client.setex(cache_key, CACHE_TTL, json.dumps(cache_data, ensure_ascii=False))
    logger.info(f"语义缓存写入：{query[:20]}")

    return results


def delete_document(source: str,tenant_id: str = "default") -> int:
    vectorstore = get_vectorstore(tenant_id)
    results = vectorstore.get(where={"source": source})
    if not results["ids"]:
        return 0
    vectorstore.delete(ids=results["ids"])

    global _bm25_cache
    _bm25_cache = [d for d in _bm25_cache if d.get("source") != source]
    with open(BM25_PATH, "w", encoding="utf-8") as f:
        json.dump(_bm25_cache, f, ensure_ascii=False)

    logger.info(f"删除文档：{source}，共 {len(results['ids'])} 个chunk")
    return len(results["ids"])


def update_document(source: str, new_text: str, metadata: dict = None, tenant_id: str = "default") -> int:
    delete_document(source, tenant_id=tenant_id)
    if metadata is None:
        metadata = {"source": source, "category": "通用"}
    return add_documents(texts=[new_text], metadatas=[metadata], tenant_id=tenant_id)