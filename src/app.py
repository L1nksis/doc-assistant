from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.agent_engine import init_engine, process_request

app = FastAPI(title="Doc Assistant Web")
engine_state = None


class ChatRequest(BaseModel):
    question: str
    mode: str = Field(default="langgraph")
    k: int = Field(default=3, ge=1, le=10)


@app.on_event("startup")
def startup():
    global engine_state
    engine_state = init_engine()


@app.get("/")
def index():
    return FileResponse("src/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    if engine_state is None:
        raise HTTPException(status_code=503, detail="Модели еще загружаются")
    try:
        return process_request(req.question, req.mode, engine_state, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
