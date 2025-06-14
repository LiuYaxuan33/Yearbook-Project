import pandas as pd
from scipy.stats import ttest_ind

# Step 1: 读取数据
df = pd.read_csv("output_/output_dpsk/deepseek_final.csv")  # 替换为你的文件路径

# Step 2: 筛选情感维度列（以 "_mean" 结尾）
emotion_cols = [col for col in df.columns if not col.endswith('_')]

# Step 3: 按性别拆分
df_male = df[df['gender_'] == 'Male']
df_female = df[df['gender_'] == 'Female']

# Step 4: 统计差异和 t 检验
results = []
for col in emotion_cols:
    male_scores = df_male[col].dropna()
    female_scores = df_female[col].dropna()
    male_mean = male_scores.mean()
    female_mean = female_scores.mean()
    diff = male_mean - female_mean
    t_stat, p_val = ttest_ind(male_scores, female_scores, equal_var=False)

    # 显著性标记
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    elif p_val < 0.1:
        sig = '.'
    else:
        sig = ''

    results.append((col.replace('_mean', ''), male_mean, female_mean, diff, p_val, sig))

# Step 5: 转换为 DataFrame
summary_df = pd.DataFrame(results, columns=['Category', 'Male Mean', 'Female Mean', 'Difference', 'p-value', 'Signif.'])
summary_df = summary_df.round(4)

# Step 6: LaTeX 表格生成函数
def generate_latex_table(df, caption="Gender Differences in Emotion Category Scores", label="tab:gender_emotion_diff"):
    latex = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrl}
\toprule
\textbf{Category} & \textbf{Male Mean} & \textbf{Female Mean} & \textbf{Difference} & \textbf{p-value} & \textbf{Signif.} \\
\midrule
"""
    for _, row in df.iterrows():
        category = row["Category"].replace('_', r'\_')
        male_mean = f"{row['Male Mean']:.4f}"
        female_mean = f"{row['Female Mean']:.4f}"
        diff = f"{row['Difference']:.4f}"
        pval = f"{row['p-value']:.4f}"
        signif = row["Signif."]
        latex += f"{category} & {male_mean} & {female_mean} & {diff} & {pval} & {signif} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
"""
    return latex

# Step 7: 输出 LaTeX 表格代码
latex_code = generate_latex_table(summary_df)
print(latex_code)
