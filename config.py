from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    CHAT_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI chat model")
    CHUNK_SIZE: int = Field(default=1000, description="Chunk size for recursive character text splitting")
    CHUNK_OVERLAP: int = Field(default=300, description="Chunk overlap for recursive character text splitting")
    RETRIEVER_K: int = Field(default=5, description="Number of documents to return from retriever")
    RETRIEVER_FETCH_K: int = Field(default=20, description="Number of documents to fetch for MMR")
    MMR_LAMBDA: float = Field(default=0.5, description="Lambda multiplier for MMR diversity (0: max diversity, 1: min diversity)")
    FAISS_INDEX_PATH: str = Field(default="./data/faiss_index", description="Path to persist FAISS index")
    SOURCE_URLS: list[str] = Field(
        default=[
            "https://stripe.com/in/privacy",
            "https://stripe.com/in/legal/privacy-center",
        ],
        description="Stripe policy documentation URLs to index"
    )

    # Evaluation Thresholds & Output Paths
    MIN_HIT_RATE: float = Field(default=0.8, description="Minimum acceptable Hit Rate @k")
    MIN_FAITHFULNESS: float = Field(default=0.9, description="Minimum acceptable Faithfulness score")
    MIN_ANSWER_RELEVANCE: float = Field(default=0.8, description="Minimum acceptable Answer Relevance score")
    MIN_CORRECTNESS: float = Field(default=0.7, description="Minimum acceptable Correctness score")
    EVAL_RESULTS_PATH: str = Field(default="./data/eval_results", description="Path to save evaluation reports")


settings = Settings()
