import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Импортируем официальные компоненты GigaChat из вашего шаблона
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Загружаем переменные окружения из .env
load_dotenv()
gigachat_creds = os.getenv("GIGACHAT_CREDENTIALS")

if not gigachat_creds:
    raise ValueError("Ошибка: Переменная GIGACHAT_CREDENTIALS не найдена в файле .env!")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================================================================
# 1. ЗАГРУЗКА ВЕКТОРНОЙ БАЗЫ (ВБ) И BERT-ЭНКОДЕРА
# ==================================================================
vector_db_path = "rag_vector_db.pkl"
if not os.path.exists(vector_db_path):
    raise FileNotFoundError(
        f"Файл '{vector_db_path}' не найден! Сначала запустите скрипт 'build_rag_index.py'."
    )

print("Загрузка векторной базы знаний...")
with open(vector_db_path, "rb") as f:
    db_data = pickle.load(f)

model_name = db_data["model_name"]
documents = db_data["documents"]
corpus_embeddings = db_data["embeddings"].to(device)

print(f"Загрузка BERT-энкодера '{model_name}'...")
bert_model = SentenceTransformer(model_name, device=device)
print(f"✅ Система готова! В базе загружено чанков: {len(documents)}")


# ==================================================================
# 2. ФУНКЦИЯ ПОИСКА TOP-K ЧАНКОВ ПО КОСИНУСНОЙ БЛИЗОСТИ
# ==================================================================
def retrieve_top_k(query, top_k=3):
    # Переводим вопрос пользователя в плотный вектор (Вопрос мой -> Э -> [плот. век])
    query_embedding = bert_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).to(device)
    
    # Считаем косинусную близость между вопросом и всеми чанками в ВБ
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Выбираем top-k лучших совпадений
    k = min(top_k, len(documents))
    top_results = torch.topk(cos_scores, k=k)
    
    retrieved = []
    for score, idx in zip(top_results.values, top_results.indices):
        doc = documents[idx.item()]
        retrieved.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": score.item()
        })
    return retrieved


# ==================================================================
# 3. СИСТЕМНАЯ ИНСТРУКЦИЯ ДЛЯ GIGACHAT (ОТВЕТ + ИСТОЧНИКИ)
# ==================================================================
system_instruction = (
    "Роль: Ты точный ИИ-ассистент системы поиска по базе знаний (RAG).\n"
    "Контекст: Тебе передается вопрос пользователя и фрагменты документов (чанки), "
    "найденные в векторной базе с помощью BERT.\n"
    "Инструкция: Ответь на вопрос пользователя на русском языке, опираясь ТОЛЬКО на предоставленный контекст. "
    "Если в контексте нет ответа, честно скажи, что в базе знаний нет этой информации. "
    "В конце ответа обязательно укажи список использованных источников.\n"
    "Формат: Структурированный, вежливый и понятный ответ."
)


# ==================================================================
# 4. ИНТЕРАКТИВНЫЙ КОНСОЛЬНЫЙ ЦИКЛ RAG
# ==================================================================
print("\n=== Консольный RAG-ассистент (BERT + GigaChat) запущен ===")
print("Задайте вопрос на русском или английском языке.")
print("Для выхода введите 'выход' или 'exit'.\n")

with GigaChat(
    base_url="https://api.giga.chat/v1",
    credentials=gigachat_creds,
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
) as client:

    while True:
        user_query = input("Ваш вопрос: ")
        if user_query.lower().strip() in ['выход', 'exit']:
            print("Завершение работы RAG-ассистента.")
            break
            
        if not user_query.strip():
            continue

        # 1. Поиск top-3 релевантных чанков в векторной базе
        top_chunks = retrieve_top_k(user_query, top_k=3)
        
        print("\n🔍 [Найденные top-3 чанка в векторной базе]:")
        context_blocks = []
        for i, item in enumerate(top_chunks, 1):
            print(f"  {i}. [{item['source']}] (близость: {item['score']:.3f}) -> {item['text'][:80]}...")
            context_blocks.append(f"Источник {i} ({item['source']}):\n{item['text']}")
            
        joined_context = "\n\n---\n\n".join(context_blocks)

        # 2. Формирование промпта: top-k чанки + Вопрос пользователя
        giga_prompt = (
            f"Контекст из базы знаний:\n{joined_context}\n\n"
            f"Вопрос пользователя: {user_query}\n\n"
            f"Сформулируй ответ на основе контекста и перечисли источники."
        )

        chat_request = Chat(
            model="GigaChat-2-Max",
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system_instruction),
                Messages(role=MessagesRole.USER, content=giga_prompt)
            ],
            temperature=0.3  # Низкая температура, чтобы GigaChat не придумывал факты вне контекста
        )

        try:
            response = client.chat(chat_request)
            print(f"\n💬 Ответ GigaChat:\n{response.choices[0].message.content}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Произошла ошибка при запросе к GigaChat API: {e}")
            print("=" * 60)
