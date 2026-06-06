import json
import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"
BM25_PATH = "./bm25_data.json"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
reranker = CrossEncoder("BAAI/bge-reranker-base")

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )


def load_bm25_data() -> list[dict]:
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_bm25_data(chunks: list[str], metadata: dict = None):
    existing = load_bm25_data()
    for chunk in chunks:
        existing.append({
            "text": chunk,
            "category": metadata.get("category", "通用") if metadata else "通用"
        })
    with open(BM25_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)

def get_bm25(category: str = None) -> tuple[BM25Okapi | None, list[str]]:
    data = load_bm25_data()
    if not data:
        return None, []
    # 按分类过滤
    if category:
        filtered = [d for d in data if d.get("category") == category]
        corpus = [d["text"] for d in filtered] if filtered else [d["text"] for d in data]
    else:
        corpus = [d["text"] for d in data]
    if not corpus:
        return None, []
    tokenized = [list(text) for text in corpus]
    return BM25Okapi(tokenized), corpus


def add_documents(texts: list[str], metadatas: list[dict] = None) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.create_documents(texts, metadatas=metadatas)

    # 把父文档的 metadata 复制到每个 chunk
    if metadatas:
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = metadatas[0]

    chunk_texts = [c.page_content for c in chunks]
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    save_bm25_data(chunk_texts, metadatas[0] if metadatas else {})
    logger.info(f"写入 {len(chunks)} 个 chunk，分类：{metadatas[0].get('category') if metadatas else '未知'}")
    return len(chunks)


def generate_queries(question: str) -> list[str]:
    prompt = f"""请将以下问题改写成3个不同的表达方式，用于检索知识库。
要求：每行一个问题，只输出问题本身，不要编号，不要解释。

原问题：{question}"""

    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    queries = [q.strip() for q in raw.split("\n") if q.strip()]
    all_queries = [question] + queries[:3]
    logger.info(f"MultiQuery 改写：{all_queries}")
    return all_queries


def rrf_fusion(vec_results: list[str], bm25_results: list[str], k: int = 60) -> list[str]:
    scores = {}
    for rank, text in enumerate(vec_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
    for rank, text in enumerate(bm25_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_results]


def search(query: str, top_k: int = 3, category: str = None) -> list[str]:
    queries = generate_queries(query)
    vectorstore = get_vectorstore()

    all_vec_results = []
    for q in queries:
        # 向量检索时加分类过滤
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
    logger.info(f"向量检索（多路去重）：{len(vec_results)} 条")

    # BM25 按分类检索
    bm25, corpus = get_bm25(category=category)
    if bm25 and corpus:
        tokenized_query = list(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k * 2]
        bm25_results = [corpus[i] for i in top_indices if bm25_scores[i] > 0]
        logger.info(f"BM25 检索：{len(bm25_results)} 条")
    else:
        bm25_results = []

    candidates = rrf_fusion(vec_results, bm25_results)[:top_k * 3]
    if not candidates:
        return []

    pairs = [[query, chunk] for chunk in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    logger.info(f"Rerank Top-{top_k}：{[(r[:15], round(float(s), 3)) for r, s in ranked[:top_k]]}")

    return [text for text, _ in ranked[:top_k]]