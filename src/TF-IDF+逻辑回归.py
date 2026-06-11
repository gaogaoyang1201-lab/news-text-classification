import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ===================== 标签ID -> 中文名称映射 =====================
id_to_name = {
    "101": "文化",
    "102": "娱乐",
    "103": "体育",
    "104": "财经",
    "106": "房产",
    "107": "汽车",
    "108": "教育",
    "109": "科技",
    "110": "军事",
    "112": "旅游",
    "113": "国际",
    "114": "股票",
    "115": "农业",
    "116": "游戏"
}

# ==========================
# 读取数据（20000条）
# ==========================
texts = []
labels = []

with open(
    "data/toutiao_cat_data.txt",
    "r",
    encoding="utf-8"
) as f:
    lines = f.readlines()

for line in lines[:20000]:
    parts = line.strip().split("_!_")
    if len(parts) >= 4:
        label = parts[1]
        text = parts[3]
        texts.append(" ".join(jieba.cut(text)))
        labels.append(label)

# 编码原始ID，用于映射中文类别
le = LabelEncoder()
le.fit(labels)
class_names = [id_to_name.get(lid, "未知") for lid in le.classes_]
print("所有分类（中文）：", class_names)

# ==========================
# 划分训练集测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)

# ==========================
# TF-IDF特征
# ==========================
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ==========================
# 训练逻辑回归
# ==========================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ==========================
# 预测
# ==========================
y_pred = model.predict(X_test)

# ==========================
# 评价指标
# ==========================
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# 关键：指定中文类别名，报告不再显示数字ID
report = classification_report(
    y_test,
    y_pred,
    target_names=class_names
)

# ==========================
# 控制台输出
# ==========================
print("Accuracy :", round(acc, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\n分类报告：")
print(report)

# ==========================
# 保存结果
# ==========================
with open(
    "report/result(TF-IDF+逻辑回归).txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("模型：TF-IDF+逻辑回归\n\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n\n")
    f.write("分类报告：\n")
    f.write(report)

print("结果已保存到 report/result(TF-IDF+逻辑回归).txt")