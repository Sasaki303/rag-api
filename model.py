from pydantic import BaseModel
from typing import List

class Dialog(BaseModel):
    role: str  # "user" または "ai"
    content: str  # 会話の内容

class DialogRequest(BaseModel):
    user_id: str  # ユーザーID
    dialogs: List[Dialog]  # 会話履歴

class ChatRequest(BaseModel):
    user_id: str  # ユーザーID
    message: str  # ユーザーの質問

class ChatResponse(BaseModel):
    user_id: str  # ユーザーID
    response: str  # AI の回答
