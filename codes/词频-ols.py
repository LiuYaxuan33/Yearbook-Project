import json
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import statsmodels.api as sm
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

filtered_person_words = [
    "able", "accurate", "active", "agreeable", "ambitious", "appropriate",
    "artistic", "athletic", "attractive", "average", "awfully", "bad",
    "bashful", "beautiful", "beautifully", "best", "better", "big", "bigger",
    "boyish", "bright", "brilliant", "busy", "calm", "capable", "certain",
    "characteristic", "charming", "cheerful", "chief", "clean", "clear",
    "confident", "conscientious", "considerate",
    "constant", "curly", "dauntless", "decidedly", "deep", "definite",
    "dependable", "determined", "devoted", "different",
    "dignified", "direct", "distinguished", "docile", "earnest", "easily",
    "easy", "easygoing", "educational", "emphatic", "energetic", "enthusiastic",
    "enviable", "equal", "especially", "evidently", "exactly", "excellent",
    "exceptional", "faithful", "faithfully", "familiar", "famous", "fast",
    "favorite", "final", "fine", "firm", "first", "fluent", "fond", "foreign",
    "former", "free", "frequently", "friendly", "full", "funny", "future",
    "generally", "genial", "gentle", "gentlemanly", "glad", "good", "great",
    "greater", "greatest", "gruff", "handy", "happy", "hard", "harder",
    "hardworking", "high", "highest", "hungry", "illustrious",
    "indispensable", "industrious", "innocent", "intelligent", "interested",
    "interesting", "invaluable", "jollier", "jolliest", "jolly", "jovial",
    "keen", "kind", "large", "last", "late", "lazy", "likable",
    "literary", "little", "live", "lively", "loyal", "mad", "major",
    "many", "merry", "mighty", "minded", "mischievous", "modest", "moody",
    "more", "most", "musically", "native", "neat", "new", "nice", "nicest",
    "noble", "noisy", "normal", "numerous", "odd", "offensive", "old",
    "only", "open", "ordinary", "original", "other", "perfect",
    "perfectly", "persistent", "phenomenal", "pleasant", "pleasing",
    "poetical", "poor", "popular", "possible", "powerful", "practical",
    "prettier", "pretty", "prime", "prominent", "proper", "proud", "pure",
    "quick", "quiet", "rare", "ready", "real", "regular", "reliable",
    "remarkable", "reserved", "reticent", "right", "sarcastic",
    "satisfied", "scandalous", "scientific", "second", "serious",
    "sharp", "short", "sincere", "slightly", "slow", "small", "smart",
    "smooth", "sociable", "social", "soft", "solemn", "splendid",
    "staunch", "steady", "stern", "stiff", "stingy", "straightforward",
    "strenuous", "strong", "stronger", "studious", "sturdy", "successful",
    "sunny", "surplus", "surprised", "tall",
    "taller", "thorough", "thoughtful","timid", "trim", "true",
    "typical", "unable", "unassuming", "unknown",
    "unlimited", "unlucky", "upper", "usually", "utter", "varied",
    "various", "very", "virtuous", "welcome", "well", "willing",
    "wise", "witty", "wonderful", "worse", "worthy", "young"
]


def custom_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ('ADJ', 'ADV')]  # 形容词 (ADJ) 和副词 (ADV)

# 1. 从 JSON 文件中加载数据
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1909_1911-1913.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. 将性别转换为二元变量：Female=1, Male=0
df['F'] = df['gender'].map({'Male': 0, 'Female': 1})

# 3. 提取评论文本，并利用 CountVectorizer 构建二值词汇矩阵
#    token_pattern 用于匹配单词，lowercase=True 自动转为小写
#vectorizer = CountVectorizer(binary=True, lowercase=True, vocabulary=filtered_person_words)
#vectorizer = CountVectorizer(binary=True, token_pattern=r'\b\w+\b', lowercase=True, min_df=1)
vectorizer = CountVectorizer(binary=True, lowercase=True, min_df=1, tokenizer=custom_tokenizer)
X = vectorizer.fit_transform(df['comment'])
vocab = vectorizer.get_feature_names_out()



# 将稀疏矩阵转换为 DataFrame
X_df = pd.DataFrame(X.toarray(), columns=vocab)

# 添加性别变量（F）
X_df['F'] = df['F'] 

# 4. 使用 OLS 回归估计每个词对“是否为女性” (F) 的影响
#    这里简化模型，没有加入课程、成绩等固定效应
X_reg = X_df.drop('F', axis=1)
X_reg = sm.add_constant(X_reg)
y = X_df['F']

model = sm.OLS(y, X_reg)
results = model.fit()
print(results.summary())

# 提取各词的回归系数、标准误及 t 统计量（剔除截距）
coefs = results.params.drop('const')
ses = results.bse.drop('const')
tstats = results.tvalues.drop('const')

# 计算 95% 置信区间
lower = coefs - 1.96 * ses
upper = coefs + 1.96 * ses

# 整理数据用于绘图
plot_df = pd.DataFrame({
    'word': coefs.index,
    'coef': coefs.values,
    't': tstats.values,
    'lower': lower.values,
    'upper': upper.values
})

# 选取 t 统计量最大的前 20 个正向词和 20 个负向词
top_positive = plot_df.sort_values('t', ascending=False).head(20)
top_negative = plot_df.sort_values('t').head(20)
plot_data = pd.concat([top_positive, top_negative]).sort_values('coef')

# 5. 绘制回归系数及 95% 置信区间图
plt.figure(figsize=(10,8))
# 这里 xerr 使用每个词对应的 1.96*标准误
plt.errorbar(plot_data['coef'], plot_data['word'], 
             xerr=1.96 * ses[plot_data['word']], fmt='o', color='blue', 
             ecolor='lightgray', capsize=3)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('regression coefficients')
plt.ylabel('words')
plt.title('Figure 1')
plt.tight_layout()
plt.show()

print(X.shape)
# 以制表符（\t）分隔的 txt 文件
X_df.to_csv("/Users/liuyaxuan/Desktop/X_output.txt", sep="\t", index=False)


correlation_matrix = pd.DataFrame(X.toarray(), columns=vocab).corr()

# 找出相关系数绝对值为 1 的变量对
perfect_collinear_vars = []
for i in range(len(vocab)):
    for j in range(i + 1, len(vocab)):  # 只需检查上三角矩阵
        if abs(correlation_matrix.iloc[i, j]) == 1:
            perfect_collinear_vars.append((vocab[i], vocab[j]))

# 打印完全共线的变量对
print("完全共线的变量对：")
for var1, var2 in perfect_collinear_vars:
    print(f"{var1} <--> {var2}")

