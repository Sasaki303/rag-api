import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# ChromaDB の設定
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "nomic-embed-text")

# 埋め込みモデルを初期化
embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL)

def get_user_collection(user_id: str):
    """ユーザーごとの ChromaDB コレクションを取得（なければ作成）"""
    collection_name = f"user_{user_id}"
    return Chroma(collection_name=collection_name, persist_directory=CHROMA_PATH, embedding_function=embedding)
