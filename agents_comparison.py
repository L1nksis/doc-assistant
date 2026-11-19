import os
import re
import ast
import json
import operator
import pickle
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters

from langchain_core.tools import tool
from langchain_gigachat import GigaChat as LangChainGigaChat
from langgraph.prebuilt import create_react_agent

load_dotenv()
gigachat_creds = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_CREDS = gigachat_creds

if not gigachat_creds:
    raise ValueError("Ошибка: Переменная GIGACHAT_CREDENTIALS не найдена в файле .env!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================================================================
# 0. ЗАГРУЗКА МОДЕЛЕЙ (RAG-БАЗА BERT И КЛАССИФИКАТОР PYTORCH)
# ==================================================================
print("Загрузка моделей для инструментов агента...")

with open("rag_vector_db.pkl", "rb") as f:
    db_data = pickle.load(f)
rag_docs = db_data["documents"]
rag_embeddings = db_data["embeddings"].to(device)
bert_model = SentenceTransformer(db_data["model_name"], device=device)

vec_path = "notebooks/vectorizer.pkl" if os.path.exists("notebooks/vectorizer.pkl") else "vectorizer.pkl"
with open(vec_path, "rb") as f:
    tfidf_vec = pickle.load(f)

enc_path = "notebooks/label_encoder.pkl" if os.path.exists("notebooks/label_encoder.pkl") else "label_encoder.pkl"
with open(enc_path, "rb") as f:
    label_enc = pickle.load(f)


class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.branch_wide = nn.Sequential(nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.4))
        self.branch_narrow = nn.Sequential(nn.Linear(input_dim, 16), nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(64 + 16, num_classes)

    def forward(self, x):
        return self.classifier(torch.cat((self.branch_wide(x), self.branch_narrow(x)), dim=1))


ckpt_path = "notebooks/model_checkpoint.pth" if os.path.exists("notebooks/model_checkpoint.pth") else "model_checkpoint.pth"
ckpt = torch.load(ckpt_path, map_location=device)
clf_model = AgentClassifierNetwork(len(tfidf_vec.get_feature_names_out()), len(label_enc.classes_)).to(device)
clf_model.load_state_dict(ckpt["model_state_dict"], strict=False)
clf_model.eval()
print("✅ Все модели успешно загружены!\n")


# ==================================================================
# 1. ТРИ ИНСТРУМЕНТА АГЕНТА И БЕЗОПАСНАЯ ОБЕРТКА run_tool
# ==================================================================
def tool_search(query: str) -> str:
    """Поиск релевантных фрагментов в базе знаний службы поддержки (RAG)."""
    q_emb = bert_model.encode(query, convert_to_tensor=True, normalize_embeddings=True).to(device)
    scores = next(iter(util.cos_sim(q_emb, rag_embeddings)))
    top_res = torch.topk(scores, k=min(2, len(rag_docs)))
    results = []
    for s, idx in zip(top_res.values, top_res.indices):
        doc = rag_docs[idx.item()]
        results.append(f"[{doc['source']} | sim={s.item():.2f}]: {doc['text']}")
    return "\n".join(results)


def tool_calc(expression: str) -> str:
    """Безопасное вычисление арифметического выражения, например '1450 * 3 - 350'."""
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Недопустимая операция в выражении")

    tree = ast.parse(expression.strip(), mode='eval')
    return str(_eval(tree.body))


def tool_classify(text: str) -> str:
    """Классификация категории обращения клиента (на английском языке)."""
    cleaned = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()
    vec = tfidf_vec.transform([cleaned]).toarray()
    tensor_x = torch.tensor(vec, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, pred = torch.max(clf_model(tensor_x), 1)
    return f"Категория обращения: {label_enc.classes_[pred.item()]}"


TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "classify": tool_classify
}


def run_tool(name: str, args) -> str:
    """Безопасная обертка вызова инструмента с обработкой ошибок."""
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS.keys())}"
    try:
        if isinstance(args, dict):
            return TOOLS[name](**args)
        return TOOLS[name](str(args))
    except Exception as e:
        return f"Ошибка выполнения инструмента '{name}': {e}"


# ==================================================================
# ВАРИАНТ 1: NATIVE TOOL CALLING (GigaChat Functions API)
# ==================================================================
NATIVE_FUNCTIONS = [
    {
        "name": "search",
        "description": "Поиск правил и инструкций в базе знаний поддержки (RAG).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос на русском или английском языке"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calc",
        "description": "Безопасное вычисление математического выражения (числа и знаки + - * /).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Арифметическое выражение, например '3 * 1450 - 350'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "classify",
        "description": "Определить категорию текста обращения клиента на английском языке.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Текст обращения клиента на английском языке"
                }
            },
            "required": ["text"]
        }
    }
]


def run_agent_native(user_query: str, max_iterations: int = 5) -> str:
    print("=" * 60)
    print("🚀 [ПОДХОД 1: NATIVE TOOL CALLING]")
    messages = [
        Messages(
            role=MessagesRole.SYSTEM,
            content="Ты ИИ-агент поддержки. Используй доступные функции для поиска информации, расчетов и классификации, затем дай финальный ответ на русском языке."
        ),
        Messages(role=MessagesRole.USER, content=user_query)
    ]

    with GigaChat(
        base_url="https://api.giga.chat/v1",
        credentials=gigachat_creds,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    ) as client:
        for step in range(max_iterations):
            chat_request = Chat(
                model="GigaChat-2-Max",
                messages=messages,
                functions=NATIVE_FUNCTIONS,
                function_call="auto",
                temperature=0.3
            )
            response = client.chat(chat_request)
            msg = next(iter(response.choices)).message
            messages.append(msg)

            if msg.function_call:
                fn_name = msg.function_call.name
                fn_args = msg.function_call.arguments
                result = run_tool(fn_name, fn_args)
                print(f"  [step {step + 1}] tool={fn_name} args={fn_args} -> {result[:100]}...")
                messages.append(
                    Messages(
                        role=MessagesRole.FUNCTION,
                        content=json.dumps({"result": result}, ensure_ascii=False)
                    )
                )
            else:
                return msg.content

    return "Превышен лимит итераций (Native)."
import time

# ==================================================================
# ВАРИАНТ 2: JSON-ПОДХОД (Ручной парсинг + extract_json)
# ==================================================================
JSON_SYSTEM_PROMPT = (
    "Роль: Ты автономный ИИ-ассистент с тремя инструментами: search, calc, classify.\n"
    "Инструкция: На каждом шаге возвращай только валидный JSON в одну строку без лишнего текста:\n"
    '{"action": "search или calc или classify или final", "input": "аргумент", "answer": "итоговый ответ"}\n'
    "Правила:\n"
    "1. Вызывай инструменты строго по очереди (по одному за шаг), только если нужных данных еще нет в истории.\n"
    "2. ЗАПРЕЩЕНО вызывать один и тот же инструмент повторно, если его результат уже получен.\n"
    "3. Когда все нужные данные собраны, сразу верни action=final и напиши полный ответ на русском языке в поле answer."
)

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("В ответе модели не найден JSON-объект")
    return json.loads(match.group(0))


def run_agent_json(user_query: str, max_steps: int = 6) -> str:
    print("=" * 60)
    print("🧩 [ПОДХОД 2: JSON-ПОДХОД]")
    messages = [
        Messages(role=MessagesRole.SYSTEM, content=JSON_SYSTEM_PROMPT),
        Messages(role=MessagesRole.USER, content=user_query)
    ]
    used_calls = set()

    with GigaChat(
        base_url="https://api.giga.chat/v1",
        credentials=gigachat_creds,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    ) as client:
        for step_num in range(max_steps):
            time.sleep(1)
            chat_request = Chat(
                model="GigaChat-2-Max",
                messages=messages,
                temperature=0.3
            )
            res = client.chat(chat_request)
            raw = next(iter(res.choices)).message.content
            print(f"  [step {step_num + 1}] raw={raw.replace(chr(10), ' ')[:140]}...")

            try:
                step_data = extract_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                messages.append(Messages(role=MessagesRole.ASSISTANT, content=raw))
                messages.append(Messages(
                    role=MessagesRole.USER,
                    content=f"Ошибка парсинга JSON ({e}). Верни строго чистый JSON."
                ))
                continue

            action = str(step_data.get("action", "")).strip().lower()
            if action == "final":
                return step_data.get("answer", "Ответ не сформирован")

            input_val = str(step_data.get("input", "")).strip()
            call_key = (action, input_val)

            if call_key in used_calls:
                messages.append(Messages(role=MessagesRole.ASSISTANT, content=raw))
                messages.append(Messages(
                    role=MessagesRole.USER,
                    content=f"Инструмент '{action}' уже вызывался! Перейди к следующему нужному инструменту или верни action='final'."
                ))
                continue

            used_calls.add(call_key)
            tool_result = run_tool(action, input_val)
            print(f"  [step {step_num + 1}] action={action} input='{input_val}' -> {tool_result[:100]}...")

            messages.append(Messages(role=MessagesRole.ASSISTANT, content=raw))
            messages.append(Messages(
                role=MessagesRole.USER,
                content=f"Результат инструмента {action}: {tool_result}\nСделай следующий шаг или верни action='final'."
            ))

    return "Превышен лимит шагов (JSON)."



# ==================================================================
# ВАРИАНТ 3: ФРЕЙМВОРК (LangGraph + LangChain)
# ==================================================================
@tool
def lc_search(query: str) -> str:
    """Поиск правил и инструкций в базе знаний поддержки (RAG)."""
    return tool_search(query)


@tool
def lc_calc(expression: str) -> str:
    """Вычислить математическое выражение (например '3 * 1250 - 500')."""
    return tool_calc(expression)


@tool
def lc_classify(text: str) -> str:
    """Определить категорию обращения клиента на английском языке."""
    return tool_classify(text)


def run_agent_framework(user_query: str) -> str:
    print("=" * 60)
    print("⚙️ [ПОДХОД 3: ФРЕЙМВОРК (LangGraph)]")
    time.sleep(1)
    llm = LangChainGigaChat(
        base_url="https://api.giga.chat/v1",
        credentials=gigachat_creds,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        model="GigaChat-2-Max",
        temperature=0.3,
    )
    agent = create_react_agent(llm, tools=[lc_search, lc_calc, lc_classify])

    final_answer = ""
    for step_idx, event in enumerate(agent.stream({"messages": [("user", user_query)]}, stream_mode="values"), 1):
        last_msg = event["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"  [step {step_idx}] LangGraph tool={tc['name']} args={tc['args']}")
        elif last_msg.type == "tool":
            print(f"  [step {step_idx}] LangGraph result -> {last_msg.content[:100]}...")
        elif last_msg.type == "ai" and last_msg.content:
            final_answer = last_msg.content

    return final_answer


# ==================================================================
# ЗАПУСК И СРАВНЕНИЕ ВСЕХ 3 ПОДХОДОВ НА СОСТАВНОМ ЗАПРОСЕ
# ==================================================================
if __name__ == "__main__":
    composite_query = (
        "Клиент написал: 'I want to cancel my order and get a refund for 3 items at 1450 rubles each minus 350 rubles delivery fee'. "
        "1) Определи категорию этого обращения на английском, "
        "2) Найди в базе знаний правила отмены заказа (cancel order), "
        "3) Посчитай точную сумму возврата."
    )

    print(f"СОСТАВНОЙ ЗАПРОС:\n{composite_query}\n")

    ans1 = run_agent_native(composite_query)
    print(f"\n💬 Финальный ответ (Native):\n{ans1}\n")

    ans2 = run_agent_json(composite_query)
    print(f"\n💬 Финальный ответ (JSON):\n{ans2}\n")

    ans3 = run_agent_framework(composite_query)
    print(f"\n💬 Финальный ответ (LangGraph):\n{ans3}\n")
