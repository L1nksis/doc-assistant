import os
import re
import ast
import json
import time
import operator
import pickle
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langchain_core.tools import tool
from langchain_gigachat import GigaChat as LangChainGigaChat
from langgraph.prebuilt import create_react_agent

load_dotenv()


class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.branch_wide = nn.Sequential(nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.4))
        self.branch_narrow = nn.Sequential(nn.Linear(input_dim, 16), nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(64 + 16, num_classes)

    def forward(self, x):
        return self.classifier(torch.cat((self.branch_wide(x), self.branch_narrow(x)), dim=1))


def init_engine():
    creds = os.getenv("GIGACHAT_CREDENTIALS")
    if not creds:
        raise ValueError("Переменная GIGACHAT_CREDENTIALS не найдена в окружении!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Загрузка векторной базы RAG
    with open("rag_vector_db.pkl", "rb") as f:
        db_data = pickle.load(f)
    rag_docs = db_data["documents"]
    rag_embeddings = db_data["embeddings"].to(device)
    bert_model = SentenceTransformer(db_data["model_name"], device=device)

    # 2. Загрузка PyTorch-классификатора
    with open("vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_enc = pickle.load(f)

    ckpt = torch.load("model_checkpoint.pth", map_location=device)
    clf_model = AgentClassifierNetwork(len(tfidf_vec.get_feature_names_out()), len(label_enc.classes_)).to(device)
    clf_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    clf_model.eval()

    return {
        "creds": creds,
        "device": device,
        "rag_docs": rag_docs,
        "rag_embeddings": rag_embeddings,
        "bert_model": bert_model,
        "tfidf_vec": tfidf_vec,
        "label_enc": label_enc,
        "clf_model": clf_model
    }


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("JSON не найден")
    return json.loads(match.group(0))


def process_request(query: str, mode: str, state: dict, k: int = 3) -> dict:
    device = state["device"]
    creds = state["creds"]
    logs = []

    # --- Инструменты ---
    def tool_search(q: str) -> str:
        q_emb = state["bert_model"].encode(q, convert_to_tensor=True, normalize_embeddings=True).to(device)
        scores = next(iter(util.cos_sim(q_emb, state["rag_embeddings"])))
        top_res = torch.topk(scores, k=min(k, len(state["rag_docs"])))
        res = []
        for s, idx in zip(top_res.values, top_res.indices):
            doc = state["rag_docs"][idx.item()]
            res.append(f"[{doc['source']} | sim={s.item():.2f}]: {doc['text']}")
        return "\n".join(res)

    def tool_calc(expr: str) -> str:
        allowed = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}
        def _ev(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                return allowed[type(node.op)](_ev(node.left), _ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](_ev(node.operand))
            raise ValueError("Недопустимая операция")
        return str(_ev(ast.parse(expr.strip(), mode='eval').body))

    def tool_classify(txt: str) -> str:
        cleaned = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', txt.lower())).strip()
        vec = state["tfidf_vec"].transform([cleaned]).toarray()
        tx = torch.tensor(vec, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, pred = torch.max(state["clf_model"](tx), 1)
        return f"Категория обращения: {state['label_enc'].classes_[pred.item()]}"

    tools_map = {"search": tool_search, "calc": tool_calc, "classify": tool_classify}

    def run_tool(name: str, args) -> str:
        if name not in tools_map:
            return f"Неизвестный инструмент: {name}"
        try:
            if isinstance(args, dict):
                val = next(iter(args.values())) if args else ""
                return tools_map[name](str(val))
            return tools_map[name](str(args))
        except Exception as e:
            return f"Ошибка: {e}"


    # --- РЕЖИМ 0: Классический RAG ---
    if mode == "rag":
        context = tool_search(query)
        logs.append(f"🔍 RAG-поиск (top-{k}): найдено {len(context.splitlines())} фрагментов")
        prompt = f"Контекст из базы знаний:\n{context}\n\nВопрос: {query}\nОтветь на русском языке строго по контексту, не используй в ответе спец символы."
        with GigaChat(base_url="https://api.giga.chat/v1", credentials=creds, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as client:
            resp = client.chat(Chat(model="GigaChat-2-Max", messages=[Messages(role=MessagesRole.USER, content=prompt)], temperature=0.3))
            return {"answer": next(iter(resp.choices)).message.content, "logs": logs}

    # --- РЕЖИМ 1: Native Tool Calling ---
    if mode == "native":
        native_funcs = [
            {"name": "search", "description": "Поиск в базе знаний поддержки (RAG).", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "calc", "description": "Вычислить математическое выражение.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
            {"name": "classify", "description": "Определить категорию обращения на английском.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
        ]
        msgs = [
            Messages(role=MessagesRole.SYSTEM, content="Ты ИИ-агент поддержки. Используй функции для поиска, расчетов и классификации, затем ответь на русском."),
            Messages(role=MessagesRole.USER, content=query)
        ]
        with GigaChat(base_url="https://api.giga.chat/v1", credentials=creds, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as client:
            for step in range(5):
                resp = client.chat(Chat(model="GigaChat-2-Max", messages=msgs, functions=native_funcs, function_call="auto", temperature=0.3))
                msg = next(iter(resp.choices)).message
                msgs.append(msg)
                if msg.function_call:
                    fn, args = msg.function_call.name, msg.function_call.arguments
                    res = run_tool(fn, args)
                    logs.append(f"🚀 [Шаг {step+1}] Native tool={fn} args={args} -> {res[:120]}...")
                    msgs.append(Messages(role=MessagesRole.FUNCTION, content=json.dumps({"result": res}, ensure_ascii=False)))
                else:
                    return {"answer": msg.content, "logs": logs}
        return {"answer": "Превышен лимит шагов.", "logs": logs}

    # --- РЕЖИМ 2: JSON-подход ---
    if mode == "json":
        sys_prompt = (
            "Ты автономный ИИ-ассистент с тремя инструментами: search, calc, classify.\n"
            "На каждом шаге возвращай только валидный JSON в одну строку без лишнего текста, не используй спец символы:\n"
            '{"action": "search или calc или classify или final", "input": "аргумент", "answer": "итоговый ответ"}\n'
            "Запрещено вызывать один и тот же инструмент повторно. Когда данные собраны, верни action=final и ответ на русском в answer."
        )
        msgs = [Messages(role=MessagesRole.SYSTEM, content=sys_prompt), Messages(role=MessagesRole.USER, content=query)]
        used = set()
        with GigaChat(base_url="https://api.giga.chat/v1", credentials=creds, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as client:
            for step in range(6):
                time.sleep(0.5)
                resp = client.chat(Chat(model="GigaChat-2-Max", messages=msgs, temperature=0.2))
                raw = next(iter(resp.choices)).message.content
                try:
                    data = extract_json(raw)
                except Exception:
                    msgs.append(Messages(role=MessagesRole.ASSISTANT, content=raw))
                    msgs.append(Messages(role=MessagesRole.USER, content="Верни строго чистый JSON."))
                    continue
                act = str(data.get("action", "")).strip().lower()
                if act == "final":
                    return {"answer": data.get("answer", ""), "logs": logs}
                inp = str(data.get("input", "")).strip()
                if (act, inp) in used:
                    msgs.append(Messages(role=MessagesRole.ASSISTANT, content=raw))
                    msgs.append(Messages(role=MessagesRole.USER, content="Этот инструмент уже вызывался! Перейди к следующему или верни final."))
                    continue
                used.add((act, inp))
                res = run_tool(act, inp)
                logs.append(f"🧩 [Шаг {step+1}] JSON action={act} input='{inp}' -> {res[:120]}...")
                msgs.append(Messages(role=MessagesRole.ASSISTANT, content=raw))
                msgs.append(Messages(role=MessagesRole.USER, content=f"Результат {act}: {res}"))
        return {"answer": "Превышен лимит шагов.", "logs": logs}

    # --- РЕЖИМ 3: LangGraph (Фреймворк) ---
    @tool
    def lc_search(q: str) -> str:
        """Поиск правил и инструкций в базе знаний поддержки (RAG)."""
        return tool_search(q)

    @tool
    def lc_calc(expr: str) -> str:
        """Вычислить математическое выражение (например '3 * 1450 - 350')."""
        return tool_calc(expr)

    @tool
    def lc_classify(txt: str) -> str:
        """Определить категорию обращения клиента на английском языке."""
        return tool_classify(txt)

    llm = LangChainGigaChat(
        base_url="https://api.giga.chat/v1",
        credentials=creds,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        model="GigaChat-2-Max",
        temperature=0.3
    )

    agent = create_react_agent(llm, tools=[lc_search, lc_calc, lc_classify])

    final_ans = ""
    for idx, ev in enumerate(agent.stream({"messages": [("user", query)]}, stream_mode="values"), 1):
        last = ev["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                logs.append(f"⚙️ [Шаг {idx}] LangGraph tool={tc['name']} args={tc['args']}")
        elif last.type == "tool":
            logs.append(f"📥 [Шаг {idx}] Результат инструмента -> {last.content[:120]}...")
        elif last.type == "ai" and last.content:
            final_ans = last.content

    return {"answer": final_ans, "logs": logs}
