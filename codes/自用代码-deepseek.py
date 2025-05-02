import openai
import pandas as pd
import json
import os
import re
import time
from tqdm import tqdm

# ✅ 初始化客户端（新版本API修改点1）
client = openai.OpenAI(
    api_key="sk-e5faf4be216d4396b8db95c40d88f574",
    base_url="https://api.deepseek.com/v1"  # DeepSeek专属端点
)

# 分析维度
dimensions = [
    "achievement", "work", "hard_working", "leadership", "social",
    "positive_emotion", "negative_emotion", "affection", "trust",
    "independence", "help", "discipline", "education"
]

# Prompt 构建
def make_prompt(comment):
    return f"""
You are an expert in psychology and linguistic analysis.  
Given the following student graduation comment:

"{comment}"

Please rate the presence of each of the following dimensions on a scale from 0 to 1 (where 0 means not present at all, 1 means strongly present).  
Output JSON with format: {{'dimension_name': score, ...}}

Dimensions:
{', '.join(dimensions)}

provide json only, without markdown citations(for example, '```json' or '```') or extra notes
"""

# 调用 DeepSeek Chat API
def analyze_comment(comment):
    try:
        response = client.chat.completions.create(  # 新API调用方式
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": make_prompt(comment)}
            ],
            temperature=0.2
        )
        text = response.choices[0].message.content  # 新属性访问方式
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"JSON 解析失败，原始响应内容: {text}")
        return {dim: 0 for dim in dimensions}
    except Exception as e:
        print(f"[Error] {e}")
        return {dim: 0 for dim in dimensions}

# 🔄 处理批次
def process_batch(students, processed_names, batch_size=100, save_path="deepseek_emotion_scores.csv"):
    results = []

    for student in tqdm(students, desc="Processing comments"):
        if student["name"] in processed_names:
            continue

        print(f"🔍 Analyzing {student['name']} ({len(processed_names)+1}/{len(students)})")
    
        comment = student["comment"]
        result = analyze_comment(comment)
        result.update({
            "name": student["name"],
            "gender": student.get("gender", ""),
            "year": student.get("year", "")
        })

        results.append(result)

        df = pd.DataFrame([result])
        df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))

        processed_names.add(student["name"])
        time.sleep(1.2)


# 📥 读取数据
with open("all_data_use.json", "r", encoding="utf-8") as f:
    students = json.load(f)

# 📂 已完成记录
processed_names = set()
save_file = "deepseek_emotion_scores.csv"
if os.path.exists(save_file):
    processed = pd.read_csv(save_file)
    processed_names = set(processed["name"].unique())

# 🔁 开始处理
process_batch(students, processed_names, batch_size=100, save_path=save_file)

# 📊 汇总并绘制雷达图
df = pd.read_csv(save_file)
meta_cols = ['name', 'gender', 'year']
emotion_cols = [col for col in df.columns if col not in meta_cols]
gender_grouped = df.groupby("gender")[emotion_cols].mean()

# 🎯 雷达图绘制
import matplotlib.pyplot as plt
import numpy as np

def plot_radar(df_grouped, title):
    labels = df_grouped.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for idx, (label, row) in enumerate(df_grouped.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# 📈 绘图
plot_radar(gender_grouped, "Gender-based Emotion Dimension Scores (DeepSeek-V3)")
