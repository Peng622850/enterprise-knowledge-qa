# 企业知识库智能问答系统

基于 RAG + Agent 架构的企业级知识库问答系统，支持多格式文档上传、智能检索、联网搜索兜底，具备完整的工程化优化措施。

## 项目演示

上传企业内部文档（PDF / Word / TXT），系统自动解析、切块、向量化存储，用户通过自然语言提问，Agent 自动决策检索策略并给出准确回答。

## 技术架构

```
用户提问
    ↓
LangGraph ReAct Agent
    ├── search_knowledge_base（知识库检索）
    │       ├── MultiQuery 改写（3路召回）
    │       ├── 向量检索（ChromaDB）
    │       ├── BM25 关键词检索
    │       ├── RRF 融合排序
    │       └── Rerank 精排（bge-reranker-base）
    └── search_web（Tavily 联网搜索兜底）
    ↓
流式输出回答
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 框架 | LangGraph（ReAct Agent） |
| 大语言模型 | DeepSeek API |
| 向量数据库 | ChromaDB |
| Embedding 模型 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Rerank 模型 | BAAI/bge-reranker-base |
| 关键词检索 | BM25（rank-bm25） |
| 联网搜索 | Tavily Search API |
| 文件解析 | pypdf + python-docx |

## RAG 工程化优化

### 召回率优化
- **混合检索**：向量检索 + BM25 关键词检索双路并行，用 RRF 算法融合结果。解决纯向量检索对专有名词、精确关键词不敏感的问题。
- **MultiQuery 多路召回**：用 LLM 将用户问题改写成 3 个不同表达方式，分别检索后合并去重。解决用户问法与文档用词不一致导致的漏召回问题。
- **文档分类过滤**：上传时为文档打分类标签，检索时按分类过滤，避免多领域文档互相干扰。

### 准确率优化
- **Rerank 重排**：向量检索召回候选集后，用 Cross-Encoder 模型（bge-reranker-base）对每个 chunk 和问题精细打分重排。相比向量相似度，Cross-Encoder 语义理解更准确。
- **相似度阈值过滤**：过滤欧氏距离大于阈值的低相关 chunk，减少噪声。
- **防幻觉 Prompt 约束**：明确要求 Agent 只根据知识库内容回答，无相关内容时如实告知，不编造。

## Agent 工程化设计

- **工具决策**：Agent 自动判断使用知识库检索还是联网搜索，知识库优先，无结果时联网兜底。
- **多轮对话记忆**：基于 LangGraph MemorySaver 实现会话级记忆，支持上下文连续对话。
- **流式输出**：Token 级别流式输出，用户无需等待全部生成完成即可看到回答。
- **迭代次数限制**：防止 Agent 陷入死循环，最大迭代次数可配置。

## 项目结构

```
knowledge_qa/
├── main.py           # FastAPI 接口层
├── agent.py          # LangGraph Agent 定义
├── rag.py            # RAG 核心逻辑（检索、向量化、BM25）
├── file_parser.py    # 文件解析（PDF / Word / TXT）
├── static/
│   └── index.html    # 前端页面
├── .env              # 环境变量（不提交）
└── requirements.txt  # 依赖列表
```

## 快速启动

**1. 克隆项目**

```bash
git clone https://github.com/Peng622850/knowledge-qa.git
cd knowledge-qa
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

**3. 配置环境变量**

新建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的DeepSeek API Key
TAVILY_API_KEY=你的Tavily API Key
```

**4. 启动服务**

```bash
uvicorn main:app --reload
```

**5. 访问系统**

浏览器打开 `http://127.0.0.1:8000`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/health` | 健康检查 |
| POST | `/upload` | 上传文件（支持分类标签） |
| POST | `/add` | 手动写入文本 |
| POST | `/chat` | 智能问答（流式输出） |

## 环境要求

- Python 3.10+
- 内存建议 8GB 以上（本地运行两个 NLP 模型）