import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from db import get_user_collection
from dotenv import load_dotenv

# 環境変数のロード
load_dotenv()

# LLM & 埋め込みモデルの初期化
llm = Ollama(model=os.getenv("LLM_MODEL", "tundere-ai:Q4_K_M"), base_url="http://127.0.0.1:11434")
embedding = OllamaEmbeddings(model=os.getenv("TEXT_EMBEDDING_MODEL", "nomic-embed-text"))

def save_dialog_summary(user_id: str, dialogs: list):
    """会話履歴を LLM で要約し、ChromaDB に保存"""
    # 🔍 ChromaDB からユーザー専用のコレクションを取得（なければ作成）
    collection = get_user_collection(user_id)

    # 📝 ユーザーの会話履歴を結合して要約
    dialog_text = "\n".join([f"{d.role}: {d.content}" for d in dialogs])
    llm_input = f"以下の会話履歴を要約し、ユーザーの特徴をリストアップしてください。\n\n{dialog_text}"
    
    print("🤖 [ログ] LLM に要約リクエスト送信...")
    summary_response = llm.invoke(llm_input)
    
    if not summary_response:
        print("⚠️ [エラー] 要約に失敗しました")
        return False

    summary_text = summary_response.strip()
    print(f"📝 [ログ] ユーザーの特徴要約: {summary_text}")

    # 🔥 **ChromaDB に保存**
    collection.add_texts([summary_text], metadatas=[{"user_id": user_id, "type": "summary"}])
    
    print("✅ [ログ] 要約データを ChromaDB に保存しました")
    return True

def generate_response(user_id: str, message: str):
    """ユーザーの質問に対して RAG で回答"""
    # 🔍 ユーザーの特徴データを ChromaDB から取得
    collection = get_user_collection(user_id)
    query_results = collection.similarity_search(message, k=3)

    if not query_results:
        print("⚠️ [エラー] ユーザーの特徴情報が見つかりませんでした。")
        context_texts = "このユーザーに関する情報はありません。"
    else:
        context_texts = "\n".join([doc.page_content for doc in query_results])

    # 🤖 **LLM にリクエスト**
    llm_input = f"以下のユーザー情報を考慮して、適切な返答をしてください。\n\n【ユーザー情報】\n{context_texts}\n\n【質問】\n{message}"
    response = llm.invoke(llm_input)

    if not response:
        print("⚠️ [エラー] AI の回答が取得できませんでした")
        return "申し訳ありません、エラーが発生しました。"

    print(f"💬 [ログ] AI の返答: {response.strip()}")
    return response.strip()
