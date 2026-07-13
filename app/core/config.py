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
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    mlflow_tracking_uri: str = "http://mlflow:5000"

    env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
