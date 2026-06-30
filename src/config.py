from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class Settings(BaseSettings):
    """
    统一配置管理中心。
    所有参数优先从 .env 文件读取，其次使用下方默认值。
    修改配置只需改 .env 文件，代码本身不需要动。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

    # ==========================================
    # LLM 提供商选择（核心开关）
    # ==========================================
    # 可选值: "ollama" | "deepseek"
    LLM_PROVIDER: Literal["ollama", "deepseek"] = "ollama"

    # ==========================================
    # Ollama 本地模型配置
    # ==========================================
    OLLAMA_MODEL: str = "qwen2:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ==========================================
    # DeepSeek API 配置
    # ==========================================
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # ==========================================
    # 通用 LLM 参数
    # ==========================================
    TEMPERATURE: float = 0.0

    # ==========================================
    # Embedding 模型（向量化，始终本地运行）
    # ==========================================
    EMBEDDING_MODEL_PATH: str = "./models/bge-small-zh-v1.5"

    # ==========================================
    # RAG 检索参数
    # ==========================================
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 4
    RERANK_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"

    # ==========================================
    # Qdrant 向量数据库
    # ==========================================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "knowledge_base"
    # bge-small-zh-v1.5 输出 512 维向量；换模型时同步修改这里
    EMBEDDING_DIM: int = 512

    # ==========================================
    # 存储路径（BM25 文档缓存，供断电重启恢复用）
    # ==========================================
    VECTORSTORE_PATH: str = "./rag_cache"

    # ==========================================
    # HyDE 检索增强
    # ==========================================
    # HyDE (Hypothetical Document Embeddings)：用 LLM 先生成"假设答案"再检索
    # 原理：问题与文档的 embedding 分布不同，用假设文档检索比用原问题更准
    # 代价：每次问答多一次 LLM 调用（约 1-3 秒）
    HYDE_ENABLED: bool = True

    # ==========================================
    # MinerU PDF 兜底（扫描件 / 复杂版式，可选）
    # ==========================================
    # 需单独安装：pip install "mineru[core]"
    # pymupdf4llm 提取字数低于 MINERU_MIN_CHARS 时自动触发
    MINERU_ENABLED: bool = True
    MINERU_MIN_CHARS: int = 100
    MINERU_BACKEND: str = "pipeline"   # CPU: pipeline；有 GPU 可试 hybrid-engine
    MINERU_METHOD: str = "auto"        # auto | txt（快）| ocr（扫描件）
    MINERU_LANGUAGE: str = "ch"
    MINERU_FORMULA: bool = False       # 关闭可提速（学术公式场景再开）
    MINERU_TABLE: bool = True          # 保留表格；纯文本 PDF 可设 false 提速
    MINERU_CLIENT_SIDE_MD: bool = True
    MINERU_START_PAGE: Optional[int] = None  # 起始页（0-based），None=全文
    MINERU_END_PAGE: Optional[int] = None    # 结束页（0-based）
    MINERU_API_URL: str = ""           # 常驻 mineru-api 地址，如 http://127.0.0.1:8000
    MINERU_TIMEOUT: int = 600

    # ==========================================
    # 网页抓取配置
    # ==========================================
    # 抓取超时（秒）
    SCRAPER_TIMEOUT: int = 15
    # 每次 /crawl 最多接受的 URL 数量，防止滥用
    SCRAPER_MAX_URLS_PER_REQUEST: int = 10

    # ==========================================
    # 翻译配置
    # ==========================================
    # 是否对抓取到的非中文网页内容自动翻译为中文
    # 翻译调用的是 LLM_PROVIDER 指定的模型，开启会增加 API 调用次数
    TRANSLATION_ENABLED: bool = False

    # ==========================================
    # LangSmith 链路追踪
    # ==========================================
    # 在 https://smith.langchain.com 注册后获取 API Key
    # 设置 LANGCHAIN_TRACING_V2=true 即可开启，不影响任何业务逻辑
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "ai-digital-employee"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ==========================================
    # 安全与系统配置
    # ==========================================
    API_KEY: str = "your-secret-key-2026"
    LOG_LEVEL: str = "INFO"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB


settings = Settings()
