import os
import re
import pickle
import pandas as pd
import numpy as np
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

# Импортируем инструменты LoRA из официальной библиотеки PEFT
from peft import LoraConfig, get_peft_model, TaskType

# Выбор устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


# ==================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА НОВЫХ ДАННЫХ ДЛЯ ДООБУЧЕНИЯ (FINE-TUNING)
# ==================================================================
print("Загрузка датасета dair-ai/emotion с Hugging Face...")
hf_dataset = load_dataset("dair-ai/emotion", "split")

# Для дообучения берем валидационную выборку (validation), чтобы данные не пересекались с train
df = pd.DataFrame(hf_dataset['validation'])

text_column = 'text' if 'text' in df.columns else df.columns
label_column = 'label' if 'label' in df.columns else df.columns[-1]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

new_cleaned_rows = []
new_labels = []
target_quantity = 500

for raw_text, raw_label in zip(df[text_column], df[label_column]):
    if len(new_cleaned_rows) >= target_quantity:
        break
    cleaned = clean_text(raw_text)
    if len(cleaned) >= 50:
        new_cleaned_rows.append(cleaned)
        # В датасете emotion метки уже идут от 0 до 5, вычитать 1 не нужно
        new_labels.append(int(raw_label))

df_fine_tune = pd.DataFrame({'cleaned_text': new_cleaned_rows, 'label': new_labels})

# Фильтрация корректных классов для 6 эмоций (0-5)
df_fine_tune = df_fine_tune.dropna(subset=['label'])
df_fine_tune = df_fine_tune[(df_fine_tune['label'] >= 0) & (df_fine_tune['label'] <= 5)].reset_index(drop=True)

# Загружаем сохраненный ранее TF-IDF векторизатор
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Векторизуем новые тексты
X_new_matrix = vectorizer.transform(df_fine_tune['cleaned_text']).toarray()
X_new_df = pd.DataFrame(X_new_matrix, columns=vectorizer.get_feature_names_out())

# Разбиваем новые 500 строк на train/test (80/20) для построения графиков
X_ft_train, X_ft_test, y_ft_train, y_ft_test = train_test_split(
    X_new_df, df_fine_tune['label'], test_size=0.2, random_state=42
)

# Определение структуры классов данных
class AgentTextDataset(Dataset):
    def __init__(self, features_df, labels_series):
        self.states = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.tensor(labels_series.values, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]


# СИНХРОНИЗАЦИЯ: Наша лучшая базовая Multi-branch архитектура, адаптированная под перехват данных от PEFT
class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=6): 
        super(AgentClassifierNetwork, self).__init__()
        
        # Первая ветвь
        self.branch_wide = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        
        # Вторая ветвь
        self.branch_narrow = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Финальный классификатор (64 + 16 = 80 признаков на входе)
        self.classifier = nn.Linear(64 + 16, num_classes)

    def forward(self, x=None, **kwargs):
        # 🔥 Перехватываем входные данные, под каким бы именем PEFT их ни передал
        if x is None and len(kwargs) > 0:
            x = next(iter(kwargs.values()))
            
        x1 = self.branch_wide(x)
        x2 = self.branch_narrow(x)
        
        out = torch.cat((x1, x2), dim=1)
        logits = self.classifier(out)
        return {"logits": logits}


# Инициализируем DataLoader'ы для дообучения
train_dataset = AgentTextDataset(X_ft_train, y_ft_train)
test_dataset = AgentTextDataset(X_ft_test, y_ft_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ==========================================
# 3. ЗАГРУЗКА И НАСТРОЙКА LORA МЕТОДОМ PEFT
# ==========================================
input_features_count = X_new_df.shape[1]  
# Инициализируем базовую модель на 6 классов эмоций
base_model = AgentClassifierNetwork(input_dim=input_features_count, num_classes=6)

# Загружаем сохраненные веса базового агента эмоций
checkpoint = torch.load('model_checkpoint.pth', map_location=device)
base_model.load_state_dict(checkpoint['model_state_dict'])
print("✅ Базовые веса модели успешно загружены!")

# Фикс для PEFT: динамически добавляем пустой объект-заглушку config
class ConfigStub:
    pass
base_model.config = ConfigStub()

# СИНХРОНИЗАЦИЯ: Конфигурация LoRA под новую архитектуру ветвей
lora_config = LoraConfig(
    r=32,                                    # Ранг
    lora_alpha=8,                            # Альфа
    lora_dropout=0.3,                        # Степень дропаута для адаптера
    # Нацеливаем LoRA на линейные слои внутри параллельных ветвей и классификатора
    target_modules=["branch_wide.0", "branch_narrow.0", "classifier"], 
    task_type=TaskType.SEQ_CLS               # Классификация текста
)

# Оборачиваем модель в LoRA-адаптер
model = get_peft_model(base_model, lora_config).to(device)
model.print_trainable_parameters()


# ==========================================
# 4. ЦИКЛ ДООБУЧЕНИЯ (FINE-TUNING) С ИСТОРИЕЙ
# ==========================================
# Оптимизатор настраиваем только на обучаемые параметры LoRA
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

print("\n--- Запуск LoRA дообучения через PEFT на новых строках ---")
ft_epochs = 20  

history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

for epoch in range(ft_epochs):
    # --- ЭТАП ОБУЧЕНИЯ ---
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for batch_states, batch_labels in train_loader:
        batch_states = batch_states.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_states)
        
        # Извлекаем логиты из словаря PEFT
        logits = outputs["logits"]
        
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_states.size(0)
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (correct_predictions / len(train_loader.dataset)) * 100
    
    # --- ЭТАП ОЦЕНКИ ---
    # Оставляем model.train(), чтобы BatchNorm корректно учитывал LoRA слои на кастомной MLP
    model.train()  
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():  # Градиенты не считаем, веса не меняются
        for batch_states, batch_labels in test_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_states)
            
            # Извлекаем логиты из словаря PEFT
            logits = outputs["logits"]
            
            loss = criterion(logits, batch_labels)
            
            val_loss += loss.item() * batch_states.size(0)
            _, predicted = torch.max(logits, 1)
            val_correct += (predicted == batch_labels).sum().item()
            
    val_epoch_loss = val_loss / len(test_loader.dataset)
    val_epoch_acc = (val_correct / len(test_loader.dataset)) * 100
    
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['test_loss'].append(val_epoch_loss)
    history['test_acc'].append(val_epoch_acc)
    
    print(f"Эпоха [{epoch+1}/{ft_epochs}] | "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
          f"Test Loss: {val_epoch_loss:.4f}, Test Acc: {val_epoch_acc:.2f}%")

# ==========================================
# 5. СОХРАНЕНИЕ ОБНОВЛЕННОГО ЧЕКПОИНТА
# ==========================================
checkpoint_path = 'model_lora_finetuned.pth'
torch.save({
    'model_state_dict': model.state_dict(),  
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': ft_epochs
}, checkpoint_path)
print(f"\n✅ Чекпоинт дообученной модели сохранен в '{checkpoint_path}'")


# ==========================================
# 6. ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================
epochs_range = range(1, ft_epochs + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], 'bo-', label='Train Loss')
plt.plot(epochs_range, history['test_loss'], 'ro-', label='Test Loss')
plt.title('Динамика Ошибки (Loss) после LoRA')
plt.xlabel('Эпохи')
plt.ylabel('Значение Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], 'b^-', label='Train Accuracy')
plt.plot(epochs_range, history['test_acc'], 'r^-', label='Test Accuracy')
plt.title('Динамика Точности (Accuracy) после LoRA')
plt.xlabel('Эпохи')
plt.ylabel('Точность (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
