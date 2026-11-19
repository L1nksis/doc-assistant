import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


# ==================================================================
# 1. ЗАГРУЗКА ДАТАСЕТА (ОБУЧЕНИЕ НА ВОПРОСАХ ИЗ 'instruction')
# ==================================================================
csv_filename = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

if not os.path.exists(csv_filename):
    raise FileNotFoundError(
        f"Файл '{csv_filename}' не найден! Положите его в папку со скриптом: {os.getcwd()}"
    )

print(f"Загрузка локального файла '{csv_filename}'...")
df = pd.read_csv(csv_filename, encoding='utf-8')

text_column = 'instruction'  # Вопросы пользователей
label_column = 'category'    # Категории обращений

# Удаляем пустые значения и перемешиваем датасет
df = df.dropna(subset=[text_column, label_column]).sample(frac=1, random_state=42).reset_index(drop=True)

# Кодируем строковые категории в числа 0..10
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df[label_column])
num_classes = len(label_encoder.classes_)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✅ Найдено классов: {num_classes}. Кодировщик сохранен в 'label_encoder.pkl'")
print(f"Используются колонки: текст -> '{text_column}', метка -> '{label_column}'")


# ==========================================
# 2. ОЧИСТКА ТЕКСТА И ОТБОР СТРОК
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
quantity_rows = 10000  # Увеличили выборку, так как вопросы короткие и обучаются быстро

for raw_text, raw_label in zip(df[text_column], df['encoded_label']):
    if len(cleaned_rows) >= quantity_rows:
        break
        
    cleaned = clean_text(raw_text)
    # Снизили порог с 50 до 15 символов, чтобы не отсекать короткие вопросы клиентов
    if len(cleaned) >= 15:
        cleaned_rows.append(cleaned)
        labels.append(int(raw_label))

cln_df = pd.DataFrame({'cleaned_text': cleaned_rows, 'label': labels})

print("\nРаспределение итоговых классов:")
for idx, class_name in enumerate(label_encoder.classes_):
    count = (cln_df['label'] == idx).sum()
    print(f"  {idx} — {class_name}: {count} строк")

cln_df.to_csv('cleaned.csv', index=False, encoding='utf-8')


# ==========================================
# 3. TF-IDF ВЕКТОРИЗАЦИЯ И СОХРАНЕНИЕ PKL
# ==========================================
print("\nИнициализация TF-IDF векторизатора...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
binary_matrix = vectorizer.fit_transform(cln_df['cleaned_text'])
feature_names = vectorizer.get_feature_names_out()

df_final = pd.DataFrame(binary_matrix.toarray(), columns=feature_names)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Векторизатор сохранен в 'vectorizer.pkl'!")


# ==========================================
# 4. РАЗДЕЛЕНИЕ НА TRAIN / TEST И DATALOADER
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df_final, cln_df['label'], test_size=0.2, random_state=42, stratify=cln_df['label']
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
# 5. АРХИТЕКТУРА НЕЙРОСЕТИ (Multi-branch MLP)
# ==========================================
class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes): 
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


# ==========================================
# 6. ИНИЦИАЛИЗАЦИЯ И ЦИКЛ ОБУЧЕНИЯ
# ==========================================
input_features_count = df_final.shape[1]
model = AgentClassifierNetwork(input_dim=input_features_count, num_classes=num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
criterion = nn.CrossEntropyLoss() 

print("\n--- Запуск процесса обучения нейросети PyTorch ---")
epochs = 8  # Для вопросов клиентов полезно сделать 8 эпох для лучшей сходимости

history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

for epoch in range(epochs):
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
    'num_classes': num_classes,
    'classes': list(label_encoder.classes_),
    'epoch': epochs
}, checkpoint_path)
print(f"\n✅ Чекпоинт сохранен в '{checkpoint_path}'")

# Сохраняем график в файл, чтобы не блокировать консоль
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
plt.savefig('training_metrics.png')
print("✅ График обучения сохранен в 'training_metrics.png'")
