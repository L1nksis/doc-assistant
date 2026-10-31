import os
import re
import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Импортируем библиотеки Hugging Face Datasets
from datasets import load_dataset

# Импортируем PyTorch библиотеки
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Подключаем библиотеку графиков
import matplotlib.pyplot as plt

# Указываем путь к предустановленным базам данных nltk
nltk.data.path.append("/usr/share/nltk_data")

# Выбор устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


# ==================================================================
# 1 и 3. ЗАГРУЗКА ДАТАСЕТА И СБОР ИСХОДНЫХ СТРОК С РЕАЛЬНЫМИ МЕТКАМИ
# ==================================================================
print("Загрузка датасета dair-ai/emotion с Hugging Face...")
# Загружаем датасет, используя конфигурацию "split"
hf_dataset = load_dataset("dair-ai/emotion", "split")

# Переводим тренировочный сплит в формат Pandas DataFrame
df = pd.DataFrame(hf_dataset['train'])

text_column = 'text' if 'text' in df.columns else df.columns[0]
label_column = 'label' if 'label' in df.columns else df.columns[-1]

print(f"Используются колонки: текст -> '{text_column}', метка -> '{label_column}'")


# ==========================================
# 2. ФУНКЦИЯ ДЛЯ ОЧИСТКИ ТЕКСТА
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


cleaned_rows = []
labels = []
quantity_rows = 5000

for raw_text, raw_label in zip(df[text_column], df[label_column]):
    if len(cleaned_rows) >= quantity_rows:
        break
        
    cleaned = clean_text(raw_text)
    if len(cleaned) >= 50:
        cleaned_rows.append(cleaned)
        # В датасете emotion метки уже находятся в диапазоне 0-5, сдвиг не требуется
        labels.append(int(raw_label))

cln_df = pd.DataFrame({'cleaned_text': cleaned_rows, 'label': labels})


# ==========================================
# 4. ПОИСК И ДОБАВЛЕНИЕ .TXT ФАЙЛОВ
# ==========================================
txt_rows = []
search_dir = "."  
txt_files_found = 0

print(f"\n--- Сканирование {search_dir} на наличие .txt файлов ---")

if os.path.exists(search_dir):
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files_found += 1
                file_full_path = os.path.join(root, file)
                file_saved_lines = 0  
                
                try:
                    with open(file_full_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            cleaned_line = clean_text(line)
                            if len(cleaned_line) >= 50:
                                txt_rows.append(cleaned_line)
                                file_saved_lines += 1
                    print(f" Найдено: '{file}' -> Сохранено строк: {file_saved_lines}")
                except Exception as e:
                    print(f" Ошибка при чтении файла '{file}': {e}")

if txt_rows:
    txt_df = pd.DataFrame(txt_rows, columns=['cleaned_text'])
    cln_df = pd.concat([cln_df, txt_df], ignore_index=True)

# 🔥 ЖЕСТКАЯ БЕЗОПАСНАЯ ФИЛЬТРАЦИЯ МЕТОК
cln_df = cln_df.dropna(subset=['label'])
cln_df['label'] = cln_df['label'].astype(int)
cln_df = cln_df[(cln_df['label'] >= 0) & (cln_df['label'] <= 5)].reset_index(drop=True)

print("\nРаспределение итоговых классов (0 — sadness, 1 — joy, 2 — love, 3 — anger, 4 — fear, 5 — surprise):")
print(cln_df['label'].value_counts())

cln_df.to_csv('cleaned.csv', index=False, encoding='utf-8')


# ==========================================
# 6. TF-IDF ВЕКТОРИЗАЦИЯ И СОХРАНЕНИЕ PKL
# ==========================================
print("\nИнициализация TF-IDF векторизатора...")
# Увеличили емкость до 3000 признаков и добавили биграммы (ngram_range=(1, 2))
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
binary_matrix = vectorizer.fit_transform(cln_df['cleaned_text'])
feature_names = vectorizer.get_feature_names_out()

df_final = pd.DataFrame(binary_matrix.toarray(), columns=feature_names)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Векторизатор сохранен в 'vectorizer.pkl'!")

# Путь адаптирован под Google Colab
output_file_csv = 'vectorized_cybersecurity_data_binary.csv'
df_final.to_csv(output_file_csv, index=False, encoding='utf-8')


# ==========================================
# 7. РАЗДЕЛЕНИЕ НА TRAIN / TEST И DATALOADER
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df_final, cln_df['label'], test_size=0.2, random_state=42
)

class AgentTextDataset(Dataset):
    def __init__(self, features_df, labels_series):
        self.states = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.tensor(labels_series.values, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]

train_dataset = AgentTextDataset(X_train, y_train)
test_dataset = AgentTextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ==========================================
# 8. АРХИТЕКТУРА НЕЙРОСЕТИ (Multi-branch MLP)
# ==========================================
class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=6): 
        super(AgentClassifierNetwork, self).__init__()
        
        # Первая ветвь: выделяет крупные, явные паттерны
        self.branch_wide = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # Вторая ветвь: жестко сжимает данные для поиска скрытых маркеров
        self.branch_narrow = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Финальный классификатор объединяет признаки из обеих ветвей (64 + 16 = 80 признаков)
        self.classifier = nn.Linear(64 + 16, num_classes)

    def forward(self, x):
        x1 = self.branch_wide(x)
        x2 = self.branch_narrow(x)
        
        # Объединяем выходы веток
        out = torch.cat((x1, x2), dim=1)
        
        return self.classifier(out)


# ==========================================
# 9. ИНИЦИАЛИЗАЦИЯ И ЦИКЛ ОБУЧЕНИЯ
# ==========================================
# Берем строго количество признаков через shape[1]
input_features_count = df_final.shape[1]  

model = AgentClassifierNetwork(input_dim=input_features_count, num_classes=6).to(device)

# Оптимизатор настроен со сбалансированным lr и сильной L2-регуляризацией (weight_decay)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
criterion = nn.CrossEntropyLoss() 

print("\n--- Запуск процесса обучения нейросети PyTorch ---")
epochs = 5  

history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

for epoch in range(epochs):
    # --- ЭТАП ОБУЧЕНИЯ ---
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for batch_states, batch_labels in train_loader:
        batch_states = batch_states.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_states)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_states.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (correct_predictions / len(train_loader.dataset)) * 100
    
    # --- ЭТАП ОЦЕНКИ ---
    model.eval()  
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():  
        for batch_states, batch_labels in test_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_states)
            loss = criterion(outputs, batch_labels)
            
            val_loss += loss.item() * batch_states.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_labels).sum().item()
            
    val_epoch_loss = val_loss / len(test_loader.dataset)
    val_epoch_acc = (val_correct / len(test_loader.dataset)) * 100
    
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['test_loss'].append(val_epoch_loss)
    history['test_acc'].append(val_epoch_acc)
    
    print(f"Эпоха [{epoch+1}/{epochs}] | "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
          f"Test Loss: {val_epoch_loss:.4f}, Test Acc: {val_epoch_acc:.2f}%")

checkpoint_path = 'model_checkpoint.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epochs
}, checkpoint_path)
print(f"\n✅ Чекпоинт сохранен в '{checkpoint_path}'")


# ==========================================
# 10. ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], 'bo-', label='Train Loss')
plt.plot(epochs_range, history['test_loss'], 'ro-', label='Test Loss')
plt.title('Динамика Ошибки (Loss) по эпохам')
plt.xlabel('Эпохи')
plt.ylabel('Значение Loss')
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], 'b^-', label='Train Accuracy')
plt.plot(epochs_range, history['test_acc'], 'r^-', label='Test Accuracy')
plt.title('Динамика Точности (Accuracy) по эпохам')
plt.xlabel('Эпохи')
plt.ylabel('Точность (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()