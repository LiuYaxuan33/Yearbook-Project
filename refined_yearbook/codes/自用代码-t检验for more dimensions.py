import pandas as pd
import statsmodels.formula.api as smf

# ——————————————————————————————————————————
# 参数设置
# ——————————————————————————————————————————
csv_path  = '/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/empath_tfidf_emotion_scores.csv'   # 你的数据文件
threshold = 0.04              # 均值阈值

# ——————————————————————————————————————————
# Step 1: 载入并清洗数据
# ——————————————————————————————————————————
df = pd.read_csv(csv_path)
df['gender'] = df['gender'].str.strip().str.capitalize()
df = df[df['gender'].isin(['Female', 'Male'])]

# ——————————————————————————————————————————
# Step 2: 提取所有情绪维度列
# ——————————————————————————————————————————
emotion_cols = [col for col in df.columns if col.lower() != 'gender']

# ——————————————————————————————————————————
# Step 3: 计算按性别分组后的均值，并筛选出任一组 ≥ 阈值的维度
# ——————————————————————————————————————————
gender_summary = df.groupby('gender')[emotion_cols].mean(numeric_only=True)
filtered_emotions = gender_summary.columns[(gender_summary > threshold).any(axis=0)].tolist()

print("▶️ 保留的维度（任一性别组均值 ≥ 0.04）：", filtered_emotions)

# ——————————————————————————————————————————
# Step 4: 对筛选后的维度做 OLS 回归，并自动提取性别系数
# ——————————————————————————————————————————
table_data = []
for col in filtered_emotions:
    model = smf.ols(f"{col} ~ C(gender)", data=df).fit()
    # 自动识别性别变量
    gender_terms = [t for t in model.params.index if t.startswith("C(gender)")]
    if gender_terms:
        term = gender_terms[0]
        coef = model.params[term]
        pval = model.pvalues[term]
    else:
        coef, pval = float('nan'), float('nan')
    r2 = model.rsquared

    # 添加星号
    if pd.isna(pval):
        stars = ''
    elif pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.1:
        stars = '*'
    else:
        stars = ''

    table_data.append({
        'Emotion':    col,
        'Coefficient': f"{coef:.3f}{stars}",
        'p-value':     f"{pval:.3f}",
        'R-squared':   f"{r2:.3f}"
    })

# ——————————————————————————————————————————
# Step 5: 输出 LaTeX 表格
# ——————————————————————————————————————————
res_df = pd.DataFrame(table_data)
latex_code = (
    res_df
    .style
    .format(na_rep='NaN')
    .to_latex(
        caption="OLS Estimates of gender Effect on Emotional Scores\n(Filtered by gender-group Mean ≥ 0.04)",
        label="tab:gender_emotion_regression",
        position="htbp"
    )
)

print("\n📄 LaTeX 表格代码：\n")
print(latex_code)
