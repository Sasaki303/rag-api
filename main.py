from fastapi import FastAPI, HTTPException
from model import DialogRequest, ChatRequest, ChatResponse
from service import save_dialog_summary, generate_response
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# FastAPI アプリケーションを作成
app = FastAPI(title="RAG Chat API", description="ChromaDB + LangChain + Ollama を活用したチャット API")

@app.post("/save_dialog", response_model=dict)
async def save_dialog(request: DialogRequest):
    """📝 ユーザーの会話履歴を LLM で要約し、ChromaDB に保存"""
    success = save_dialog_summary(request.user_id, request.dialogs)
    if not success:
        raise HTTPException(status_code=500, detail="要約データの保存に失敗しました")
    return {"message": "会話要約を保存しました"}

@app.post("/ask", response_model=ChatResponse)
async def ask(request: ChatRequest):
    """🤖 ユーザーの質問に対して RAG で回答"""
    response = generate_response(request.user_id, request.message)
    return ChatResponse(user_id=request.user_id, response=response)
