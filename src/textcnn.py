import jieba
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================
# 读取数据（20000条）
# ==========================
texts = []
labels = []

with open("data/toutiao_cat_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines[:20000]:
    parts = line.strip().split("_!_")
    if len(parts) >= 4:
        label = parts[1]
        title = parts[3]
        # 结巴分词
        words = jieba.cut(title)
        text = " ".join(words)
        texts.append(text)
        labels.append(label)

# ==========================
# 标签编码
# ==========================
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ==========================
# 构建词表 & 文本序列转换
# ==========================
word2idx = {}
idx = 1
for text in texts:
    for word in text.split():
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1
print("词表大小：", len(word2idx))

max_len = 30
X = []
for text in texts:
    seq = []
    for word in text.split():
        seq.append(word2idx.get(word, 0))
    seq = seq[:max_len]
    # 补0到固定长度
    while len(seq) < max_len:
        seq.append(0)
    X.append(seq)

# 转为Tensor
X = torch.tensor(X)
y = torch.tensor(labels)

# ==========================
# 划分训练集测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# TextCNN 模型定义
# ==========================
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(word2idx) + 1,
            embedding_dim=128
        )
        self.conv = nn.Conv2d(1, 100, (3, 128))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(max_len - 3 + 1)
        self.fc = nn.Linear(100, len(set(labels.tolist())))

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.squeeze(3)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x

# ==========================
# 模型、损失、优化器
# ==========================
model = TextCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================
# 模型训练
# ==========================
print("开始训练 TextCNN...")
epochs = 5
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ==========================
# 预测 & 评价指标（与SVM代码对齐）
# ==========================
with torch.no_grad():
    outputs = model(X_test)
    _, y_pred = torch.max(outputs, 1)

# 转回numpy用于sklearn指标计算
y_test_np = y_test.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()

acc = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np, average="weighted")
recall = recall_score(y_test_np, y_pred_np, average="weighted")
f1 = f1_score(y_test_np, y_pred_np, average="weighted")
report = classification_report(y_test_np, y_pred_np)

# ==========================
# 控制台输出
# ==========================
print("\n========== TextCNN 评价结果 ==========")
print("Accuracy :", round(acc, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\n分类报告：")
print(report)

# ==========================
# 保存结果（路径、格式与SVM统一）
# ==========================
with open(
    "report/result(textcnn_epoch5).txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("模型：TextCNN\n\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n\n")
    f.write("分类报告：\n")
    f.write(report)

print("\n结果已保存到 report/result(textcnn_epoch5).txt")