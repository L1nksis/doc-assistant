import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

load_dotenv()


def build_index(vector_db_path="rag_vector_db.pkl"):
    """Загружает векторную базу, BERT-энкодер и проверяет токен GigaChat при старте сервера."""
    if not os.path.exists(vector_db_path):
        raise FileNotFoundError(
            f"Файл '{vector_db_path}' не найден в {os.getcwd()}! Сначала соберите индекс."
        )

    gigachat_creds = os.getenv("GIGACHAT_CREDENTIALS")
    if not gigachat_creds:
        raise ValueError("Переменная GIGACHAT_CREDENTIALS не найдена в файле .env!")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(vector_db_path, "rb") as f:
        db_data = pickle.load(f)

    model_name = db_data["model_name"]
    documents = db_data["documents"]
    corpus_embeddings = db_data["embeddings"].to(device)

    bert_model = SentenceTransformer(model_name, device=device)

    return {
        "device": device,
        "bert_model": bert_model,
        "documents": documents,
        "corpus_embeddings": corpus_embeddings,
        "gigachat_creds": gigachat_creds
    }


def rag_answer(question: str, index_state: dict, k: int = 3):
    """Ищет top-k чанков и генерирует ответ через GigaChat API."""
    device = index_state["device"]
    bert_model = index_state["bert_model"]
    documents = index_state["documents"]
    corpus_embeddings = index_state["corpus_embeddings"]
    gigachat_creds = index_state["gigachat_creds"]

    # 1. Векторизация вопроса и поиск top-k по косинусному сходству
    query_embedding = bert_model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).to(device)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(k, len(documents))
    top_results = torch.topk(cos_scores, k=top_k)

    context_blocks = []
    sources = []

    for score, idx in zip(top_results.values, top_results.indices):
        doc = documents[idx.item()]
        source_info = f"{doc['source']} (близость: {score.item():.3f})"
        sources.append(source_info)
        context_blocks.append(f"Источник ({doc['source']}):\n{doc['text']}")

    joined_context = "\n\n---\n\n".join(context_blocks)

    # 2. Запрос к GigaChat API строго по вашему шаблону
    system_instruction = (
        "Роль: Ты точный ИИ-ассистент системы поиска по базе знаний (RAG).\n"
        "Инструкция: Ответь на вопрос пользователя на русском языке, опираясь ТОЛЬКО на предоставленный контекст. "
        "Если в контексте нет ответа, честно скажи, что в базе знаний нет этой информации."
    )

    giga_prompt = (
        f"Контекст из базы знаний:\n{joined_context}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        f"Сформулируй ответ на основе контекста."
    )

    chat_request = Chat(
        model="GigaChat-2-Max",
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system_instruction),
            Messages(role=MessagesRole.USER, content=giga_prompt)
        ],
        temperature=0.3
    )

    with GigaChat(
        base_url="https://api.giga.chat/v1",
        credentials=gigachat_creds,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    ) as client:
        response = client.chat(chat_request)
        answer = response.choices[0].message.content

    return answer, sources
