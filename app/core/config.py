from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://ekie:ekie_password@postgres:5432/ekie"
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ekie_password"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "ekie_documents"
    google_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    gemini_timeout_seconds: float = 30.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many fused hybrid-search candidates to feed the cross-encoder
    # before cutting down to top_k. Reranking a wider pool than top_k is
    # the whole point (RRF's top-k is a recall list, not a precision list) —
    # too small a pool and the reranker never sees the chunk it would have
    # ranked #1; too large and re-scoring gets slow since the cross-encoder
    # is O(pool_size) forward passes, not a single vector comparison.
    rerank_pool_size: int = 30
    # Hard ceiling on rerank_pool_size regardless of what a caller (or the
    # /search/hybrid query param) requests. Without this, an unbounded
    # rerank_pool_size forces semantic_search()/keyword_search() to fetch
    # that many rows each and then runs that many synchronous cross-encoder
    # forward passes per request — a cheap way to stall the API if this
    # endpoint is ever reachable by anyone other than a trusted caller.
    rerank_pool_size_max: int = 200
    # Hard timeout around the cross-encoder's predict() call, mirroring
    # gemini_timeout_seconds — a stalled model load or an unusually large
    # pool shouldn't be able to hang a request indefinitely.
    reranker_timeout_seconds: float = 10.0
    mlflow_tracking_uri: str = "http://mlflow:5000"
    env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()