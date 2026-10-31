import os
import re
import pickle
import torch
import torch.nn as nn
from dotenv import load_dotenv

# Импортируем официальные компоненты GigaChat из вашего шаблона
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Загружаем переменные окружения из .env
load_dotenv()
gigachat_creds = os.getenv("GIGACHAT_CREDENTIALS")

if not gigachat_creds:
    raise ValueError("Ошибка: Переменная GIGACHAT_CREDENTIALS не найдена в файле .env!")

# Выбор устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================================================================
# 1. ЗАГРУЗКА ЛОКАЛЬНЫХ ИИ-КОМПОНЕНТОВ (ВЕКТОРИЗАТОР И АГЕНТ)
# ==================================================================
print("Загрузка компонентов ИИ-агента...")

try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    input_features_count = len(vectorizer.get_feature_names_out())
except FileNotFoundError:
    raise FileNotFoundError("Ошибка: Не найден файл 'vectorizer.pkl'. Пожалуйста, перенесите его в эту папку.")

EMOTION_MAP = {
    0: "sadness (грусть) 😢",
    1: "joy (радость) 😊",
    2: "love (любовь) ❤️",
    3: "anger (злость) 😡",
    4: "fear (страх) 😨",
    5: "surprise (удивление) 😲"
}

class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=6): 
        super(AgentClassifierNetwork, self).__init__()
        self.branch_wide = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.branch_narrow = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(64 + 16, num_classes)

    def forward(self, x):
        x1 = self.branch_wide(x)
        x2 = self.branch_narrow(x)
        out = torch.cat((x1, x2), dim=1)
        return self.classifier(out)

model = AgentClassifierNetwork(input_dim=input_features_count, num_classes=6).to(device)
checkpoint_path = 'model_lora_finetuned.pth'
if not os.path.exists(checkpoint_path):
    checkpoint_path = 'model_checkpoint.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("base_model.model.", "")
        cleaned_state_dict[name] = v
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    print(f"✅ Веса агента успешно загружены из '{checkpoint_path}'!")
except FileNotFoundError:
    raise FileNotFoundError("Ошибка: Не найдены чекпоинты модели. Сначала обучите модель.")


# ==================================================================
# 2. НАСТРОЙКА ИНСТРУКЦИЙ И ПРЕДОБРАБОТКИ
# ==================================================================
system_instruction = (
    "Роль: Ты эмпатичный ИИ-ассистент и эксперт по эмоциональному интеллекту.\n"
    "Контекст: Пользователь ввел фразу на английском языке, а локальный классификатор ИИ определил её эмоцию.\n"
    "Инструкция: Проанализируй фразу пользователя и выданный вердикт. Напиши краткий, "
    "но развернутый ответ на русском языке: поддержи пользователя, если эмоция негативная, "
    "или порадуйся за него, если позитивная. Дай один уместный жизненный совет или комментарий по ситуации.\n"
    "Формат: Отвечай дружелюбно, структурированно, без использования программных логов."
)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==================================================================
# 3. ИНТЕРАКТИВНЫЙ КОНСОЛЬНЫЙ ЦИКЛ ОПРОСА
# ==================================================================
print("\n=== Консольный интерфейс ИИ-агента запущен ===")
print("Введите вашу фразу на английском языке.")
print("Для выхода введите 'выход' или 'exit'.\n")

# Используем контекстный менеджер 'with' для стабильности сессии GigaChat на Windows
with GigaChat(
    base_url="https://api.giga.chat/v1",
    credentials=gigachat_creds,
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
) as client:

    while True:
        user_input = input("Вы (English): ")
        if user_input.lower().strip() in ['выход', 'exit']:
            print("Завершение работы консоли.")
            break
            
        if not user_input.strip():
            continue

        # 1. Локальная предобработка текста
        cleaned_text_sample = clean_text(user_input)
        if len(cleaned_text_sample) < 10:
            print("🤖 Агент: Фраза слишком короткая для надежного определения эмоции.")
            print("-" * 50)
            continue

        # 2. Векторизация и классификация локальной моделью PyTorch
        vectorized_sample = vectorizer.transform([cleaned_text_sample]).toarray()
        tensor_sample = torch.tensor(vectorized_sample, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(tensor_sample)
            _, predicted_class = torch.max(outputs, 1)
            predicted_idx = predicted_class.item()
        
        detected_emotion = EMOTION_MAP.get(predicted_idx, 'Неизвестная эмоция')
        print(f"🧠 [Локальный агент определил класс]: {detected_emotion}")

        # 3. Формирование запроса по вашему SDK-шаблону
        giga_prompt = (
            f"Фраза пользователя: \"{user_input}\"\n"
            f"Определенная локальным агентом эмоция: {detected_emotion}\n"
            f"Сформулируй свой финальный ответ-комментарий."
        )
        
        chat_request = Chat(
            model="GigaChat-2-Max",
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system_instruction),
                Messages(role=MessagesRole.USER, content=giga_prompt)
            ],
            temperature=0.7
        )

        try:
            # Отправка запроса через официальный метод SDK
            response = client.chat(chat_request)
            print(f"\n💬 Ответ GigaChat:\n{response.choices[0].message.content}")
            print("-" * 50)

        except Exception as e:
            print(f"❌ Произошла ошибка при запросе к GigaChat API: {e}")
            print("-" * 50)
