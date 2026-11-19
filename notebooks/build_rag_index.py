import os
import pickle
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Выбор устройства (GPU или CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")


# ==================================================================
# 1. ФУНКЦИЯ ЧАНКИНГА (КАК НА ДОСКЕ: 500 СИМВОЛОВ, ПЕРЕКРЫТИЕ 50)
# ==================================================================
def split_into_chunks(text, chunk_size=500, overlap=50):
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Убираем лишние переносы строк, сохраняя структуру текста
    text = " ".join(text.split())
    chunks = []
    start = 0
    step = chunk_size - overlap  # Шаг сдвига: 500 - 50 = 450
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= 30:  # Отсекаем совсем короткие обрывки
            chunks.append(chunk)
        start += step
        
    return chunks


# ==================================================================
# 2. СБОР ДОКУМЕНТОВ ИЗ .TXT ФАЙЛОВ И CSV-ДАТАСЕТА
# ==================================================================
documents = []  # Список словарей: {'text': чанк, 'source': имя_источника}

# --- А. Поиск локальных .txt файлов в текущей папке ---
search_dir = "."
print(f"\n--- Сканирование '{os.path.abspath(search_dir)}' на наличие .txt файлов ---")

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                file_chunks = split_into_chunks(full_text, chunk_size=500, overlap=50)
                for idx, ch in enumerate(file_chunks):
                    documents.append({
                        'text': ch,
                        'source': f"Файл: {file} (чанк #{idx + 1})"
                    })
                print(f" ✅ Файл '{file}' -> создано чанков: {len(file_chunks)}")
            except Exception as e:
                print(f" ❌ Ошибка чтения '{file}': {e}")

# --- Б. Подгрузка ответов из CSV-датасета (если файл лежит рядом) ---
csv_filename = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
USE_CSV_DATA = True      # Поставьте False, если хотите искать ТОЛЬКО по .txt файлам
MAX_CSV_ROWS = 1000      # Сколько строк взять из CSV для базы знаний

if USE_CSV_DATA and os.path.exists(csv_filename):
    print(f"\nЗагрузка базы знаний из '{csv_filename}' (первые {MAX_CSV_ROWS} уникальных ответов)...")
    df = pd.read_csv(csv_filename, encoding='utf-8').dropna(subset=['response', 'category'])
    df = df.drop_duplicates(subset=['response']).head(MAX_CSV_ROWS)
    
    csv_chunks_count = 0
    for idx, row in df.iterrows():
        resp_chunks = split_into_chunks(row['response'], chunk_size=500, overlap=50)
        for ch in resp_chunks:
            documents.append({
                'text': ch,
                'source': f"База поддержки [{row['category']}] (запись #{idx})"
            })
            csv_chunks_count += 1
    print(f" ✅ Из CSV добавлено чанков: {csv_chunks_count}")

if not documents:
    raise ValueError("База знаний пуста! Добавьте хотя бы один .txt файл или CSV-датасет в папку со скриптом.")

print(f"\nВсего подготовлено чанков для индексации: {len(documents)}")


# ==================================================================
# 3. ВЕКТОРИЗАЦИЯ ЧЕРЕЗ BERT И СОХРАНЕНИЕ ВЕКТОРНОЙ БАЗЫ (ВБ)
# ==================================================================
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"\nЗагрузка BERT-модели '{model_name}'...")
bert_model = SentenceTransformer(model_name, device=device)

texts_to_encode = [doc['text'] for doc in documents]
print("Кодирование чанков в плотные векторы (эмбеддинги)...")

# normalize_embeddings=True сразу нормирует векторы для быстрого косинусного сходства
embeddings = bert_model.encode(
    texts_to_encode,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
).cpu()

# Сохраняем векторную базу (ВБ) на диск
vector_db_path = "rag_vector_db.pkl"
with open(vector_db_path, "wb") as f:
    pickle.dump({
        "model_name": model_name,
        "documents": documents,
        "embeddings": embeddings
    }, f)

print(f"\n✅ Векторная база успешно сохранена в '{vector_db_path}'! Размерность матрицы: {embeddings.shape}")
