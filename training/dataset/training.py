import re
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

csv_path = Path(__file__).resolve().parent / "synthetic_logs.csv"
df = pd.read_csv(csv_path)
# print(df.head())
# print(df.source.unique())
# print(df.target_label.unique())
# print(df[df.target_label=='System Notification'].sample(10))
# print(df[df.log_message.str.startswith("System reboot initiated by user")])


# ============================================================
# 第一步：DBSCAN 聚类
# 先用 Sentence Transformer 把所有日志映射到向量空间，
# 再用 DBSCAN 计算向量之间的相似度进行聚类，
# 最后按 cluster 包含的日志数目从多到少排序
# ============================================================
# 旧方案：TF-IDF + DBSCAN（已改用 Sentence Transformer）
# from sklearn.feature_extraction.text import TfidfVectorizer
# log_messages = df['log_message']#extract log messages
# vectorizer = TfidfVectorizer(stop_words='english')#convert log messages to TF-IDF features
# X = vectorizer.fit_transform(log_messages)
# dbscan = DBSCAN(eps=0.5, min_samples=5, metric ='cosine')
# clusters = dbscan.fit_predict(X)
# print(clusters)
model = SentenceTransformer('all-MiniLM-L6-v2')#Load pre-trained SentenceTransformer model
embeddings = model.encode(df['log_message'].tolist())#Generate embeddings for log messages
# print(embeddings[:5])
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')#Perform DBSCAN clustering
clusters = dbscan.fit_predict(embeddings)
df['cluster'] = clusters#Add cluster labels to the DataFrame
print(df.head())
clusters = df.groupby("cluster")["log_message"].apply(list)
sorted_clusters = clusters.sort_values(key = lambda x: x.map(len),ascending=False)


# ============================================================
# 第二步：提取包含日志最多的前几个 cluster
# ============================================================
print("Clustered Patterns:")
for cluster_id, messages in sorted_clusters.items():
    if len(messages) > 10:
        print(f"Cluster {cluster_id}:")
        for msg in messages[:5]:
            print(f"  {msg}")


# ============================================================
# 第三步：根据这些高频 cluster 写 Regex，提取关键信息并分类
# ============================================================
def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message, re.IGNORECASE):
            return label
    return None
# print(classify_with_regex("User User123 logged out."))
df['regex_label'] = df['log_message'].apply(classify_with_regex)
# print(df[df.regex_label.isna()])
df_non_regex = df[df['regex_label'].isnull()].copy()
# print(df_non_regex.shape)


# ============================================================
# 第四步：剔除样本太少的数据（模型学不会）
# 这里剔除的是 LegacyCRM 来源的日志，这部分线上交给大模型处理
# ============================================================
df_legacy = df_non_regex[df_non_regex.source=="LegacyCRM"]
# print(df_legacy)
df_non_legacy = df_non_regex[df_non_regex.source!="LegacyCRM"]
# print(df_non_legacy)


# ============================================================
# 第五步：用 Sentence Transformer 把剩余日志（以句为单位）映射到向量空间
# ============================================================
embeddings_filtered = model.encode(df_non_legacy['log_message'].tolist())
# print(len(embeddings_filtered))
X = embeddings_filtered
y = df_non_legacy['target_label'].astype(str).to_numpy()


# ============================================================
# 第六步：训练 Logistic Regression 并保存模型
# 线上使用时：用 clf.predict_proba() 得到置信度，
# 置信度小于 0.5 的 log 直接交给大模型分类
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
# print(report)

model_dir = Path(__file__).resolve().parent.parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "log_classifier.joblib"
joblib.dump(clf, model_path)
