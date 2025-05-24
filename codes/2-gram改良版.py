import json
import os
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
BLACKLIST = {}  # 自定义黑名单词
USE_1GRAM = False
MIN_DF = 5  # 最小文档频率阈值
print(1)
# 1. 预处理函数：直接生成有效2-gram


def old_generate_features(texts):
    for doc in nlp.pipe(texts, batch_size=50):
        tokens = []
        pos_tags = []
        
        # 第一步：过滤停用词、标点、黑名单
        for token in doc:
            if (not token.is_stop
                and not token.is_punct 
                and token.text.lower() not in BLACKLIST):
                tokens.append(token.text.lower())
                pos_tags.append(token.pos_)
        
        # 第二步：生成有效特征（1-gram和2-gram）
        valid_terms = []
        
        # 生成1-gram（仅ADJ/ADV）
        if USE_1GRAM:
            for i in range(len(tokens)):
                if pos_tags[i] in {'ADJ', 'ADV'}:
                    valid_terms.append(tokens[i])  # 1-gram格式
        
        # 生成2-gram（至少一个ADJ/ADV）
        for i in range(len(tokens)-1):
            if pos_tags[i] in {'ADJ', 'ADV'} or pos_tags[i+1] in {'ADJ', 'ADV'}:
                valid_terms.append(f"{tokens[i]}_{tokens[i+1]}")  # 2-gram格式
        
        yield ' '.join(valid_terms)

def generate_features(texts):
    for doc in nlp.pipe(texts, batch_size=50):
        # 先构造一个布尔数组，标记每个 token 是否“有效”
        valid_mask = [
            (not token.is_stop
             and not token.is_punct
             and token.text.lower() not in BLACKLIST)
            for token in doc
        ]
        
        # 构造 tokens_lower 与 pos_tags 两个列表，仅用于 1-gram
        tokens_lower = [token.text.lower() for token, valid in zip(doc, valid_mask) if valid]
        pos_tags =       [token.pos_       for token, valid in zip(doc, valid_mask) if valid]
        
        # 第一步：生成1-gram（仅 ADJ/ADV）
        valid_terms = []
        if USE_1GRAM:
            for tok, pos in zip(tokens_lower, pos_tags):
                if pos in {'ADJ', 'ADV'}:
                    valid_terms.append(tok)
        
        # 第二步：生成2-gram
        # 基于原始 doc 中的相邻 token 对
        for i in range(len(doc) - 1):
            t1, t2 = doc[i], doc[i+1]
            # 两个词必须都是“有效”的
            if not (valid_mask[i] and valid_mask[i+1]):
                continue
            # 至少一个是 ADJ/ADV/NOUN
            if t1.pos_ in {'ADJ', 'ADV', 'NOUN'} or t2.pos_ in {'ADJ', 'ADV', 'NOUN'}:
                bigram = f"{t1.text.lower()}_{t2.text.lower()}"
                valid_terms.append(bigram)
        
        yield ' '.join(valid_terms)


print(2)
# 2. 加载数据
with open('all_data_use_labeled.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df['F'] = df['gender'].map({'Male': 1, 'Female': 0})

# 3. 生成预处理后的文本（包含有效2-gram）
df['processed'] = list(generate_features(df['comment']))
print(3)
# 4. 构建特征矩阵
vectorizer = CountVectorizer(
    binary=True,
    tokenizer=lambda x: x.split(),  # 按空格分割预处理后的字符串
    lowercase=False,  # 预处理阶段已统一小写
    min_df=MIN_DF
)
X = vectorizer.fit_transform(df['processed'])
y = df['F']
# 输出稀疏矩阵维度
print("示例特征:", vectorizer.get_feature_names_out()[:5])
print("Feature matrix shape:", X.shape)
print(4)
# 5. Lasso回归（后续代码保持不变）
lasso_cv = LassoCV(cv=5, max_iter=10000)
print(4.3)
lasso_cv.fit(X, y)
print(4.5)
best_alpha = lasso_cv.alpha_
lasso = Lasso(alpha=best_alpha, max_iter=10000)
print(4.8)
lasso.fit(X, y)

print(5)
# 用最佳 alpha 重新训练模型
best_alpha = lasso_cv.alpha_
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X, y)
print(6)
coefs = lasso.coef_
selected_features = np.where(coefs != 0)[0]
vocab_selected = vectorizer.get_feature_names_out()[selected_features]
coefs_selected = coefs[selected_features]
print(7)
# 创建完整的结果表格并按系数排序
result_df = pd.DataFrame({
    'word': vocab_selected,
    'coefficient': coefs_selected
}).sort_values('coefficient', ascending=False)  # 按系数从大到小排序
# 输出表格到CSV文件
result_df.to_csv("2-grams.csv", index=False)

# 控制台打印结果
print("\n非零系数特征列表：")
print(result_df.to_string(index=False))




# 整理数据用于绘图
plot_df = pd.DataFrame({
    'word': vocab_selected,
    'coef': coefs_selected
})


# 5. 绘制回归系数图（Lasso 无标准误信息）

# 1. 按系数排序
df_sorted = result_df.sort_values('coefficient')

# 2. 动态计算图高
n_items = len(df_sorted)
fig_height = max(8, n_items * 0.3)

save_dir = './output_/output_basic/'
# 确保目录存在
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, '2-grams.png')


# 3. 绘图
plt.figure(figsize=(8, fig_height))
plt.scatter(df_sorted['coefficient'], df_sorted['word'], s=50)
plt.axvline(x=0, linestyle='--')
plt.grid(axis='x', linestyle='--', linewidth=0.5)

# 4. 标注与样式
plt.xlabel('Lasso Regression Coefficients', fontsize=12)
plt.ylabel('Words', fontsize=12)
plt.title(f'Figure 1 (Best alpha={best_alpha:.4f})', fontsize=14)
plt.yticks(fontsize=8)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'图已保存到：{save_path}')

