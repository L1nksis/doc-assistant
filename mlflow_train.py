import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# ==================================================================
# 1. ПОДГОТОВКА ДАННЫХ (ОБУЧЕНИЕ НА ВОПРОСАХ ИЗ 'instruction')
# ==================================================================
csv_filename = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
if not os.path.exists(csv_filename):
    csv_filename = os.path.join("notebooks", csv_filename)

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"Файл датасета не найден! Проверьте путь: {csv_filename}")

df = pd.read_csv(csv_filename, encoding='utf-8')
df = df.dropna(subset=['instruction', 'category']).sample(frac=1, random_state=42).reset_index(drop=True)

label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['category'])
num_classes = len(label_encoder.classes_)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9а-яё\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


cleaned_rows, labels = [], []
for raw_text, raw_label in zip(df['instruction'], df['encoded_label']):
    if len(cleaned_rows) >= 6000:
        break
    cleaned = clean_text(raw_text)
    if len(cleaned) >= 15:
        cleaned_rows.append(cleaned)
        labels.append(int(raw_label))

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
X_matrix = vectorizer.fit_transform(cleaned_rows).toarray()
y_array = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X_matrix, y_array, test_size=0.2, random_state=42, stratify=y_array
)

input_features_count = X_train.shape[1]


class AgentTextDataset(Dataset):
    def __init__(self, X, y):
        self.states = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]


train_loader = DataLoader(AgentTextDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(AgentTextDataset(X_test, y_test), batch_size=32, shuffle=False)


# ==================================================================
# 2. АРХИТЕКТУРА НЕЙРОСЕТИ PYTORCH
# ==================================================================
class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
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
        return self.classifier(torch.cat((self.branch_wide(x), self.branch_narrow(x)), dim=1))


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return round((correct / total) * 100, 2)


# ==================================================================
# 3. ИНИЦИАЛИЗАЦИЯ MLFLOW И ФУНКЦИЯ ОБУЧЕНИЯ (ЗАДАНИЕ №1)
# ==================================================================
mlflow.set_experiment("doc-classifier")


def train_once(lr, epochs):
    run_name = f"mlp_lr_{lr}_ep_{epochs}"
    with mlflow.start_run(run_name=run_name):
        # Логируем гиперпараметры и теги
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("max_features", input_features_count)
        mlflow.set_tag("model_type", "MultiBranchMLP")
        mlflow.set_tag("dataset", "Bitext-Customer-Support")

        model = AgentClassifierNetwork(input_dim=input_features_count, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        train_losses, test_accs = [], []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = evaluate(model, test_loader)
            train_losses.append(epoch_loss)
            test_accs.append(epoch_acc)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
            mlflow.log_metric("test_accuracy", epoch_acc, step=epoch + 1)

        final_acc = test_accs[-1]
        print(f"Запуск [{run_name}] завершен -> Final Test Accuracy: {final_acc}%")

        # Сохраняем и логируем график как дополнительный артефакт
        plot_path = "training_curve.png"
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, 'bo-', label='Train Loss')
        plt.title(f'Loss (lr={lr})')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), test_accs, 'r^-', label='Test Acc (%)')
        plt.title(f'Accuracy (lr={lr})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        # Логируем файл конфигурации классов как второй артефакт
        config_path = "classes_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"num_classes": num_classes, "classes": list(label_encoder.classes_)}, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(config_path)

        # Передаем 2 строки из X_test как пример входа (требуется для BatchNorm1d и pt2)
        model.eval()
        sample_input = X_test[:2].astype(np.float32)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=sample_input
        )

        return final_acc


# ==================================================================
# 4. СЕРИЯ ЗАПУСКОВ ДЛЯ ПОДБОРА ГИПЕРПАРАМЕТРОВ (4 ЭКСПЕРИМЕНТА)
# ==================================================================
if __name__ == "__main__":
    print("Запуск серии экспериментов MLflow...")
    configs = [
        (0.0001, 5),
        (0.0005, 8),
        (0.001, 10),
        (0.01, 5),
    ]
    for lr, epochs in configs:
        train_once(lr, epochs)

    print("\n✅ Все эксперименты завершены! Запустите 'mlflow ui' в консоли.")
