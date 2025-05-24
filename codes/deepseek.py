import openai
import pandas as pd
import json
import os
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np


load_dotenv()  # 加载.env文件中的环境变量



# ✅ 初始化客户端（新版本API修改点1）
client = openai.OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"  # DeepSeek专属端点
)

# 分析维度
with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
NEW_WORDS = dimension_all[1]

CUSTOM_CATEGORIES = []

for key, value in framework.items():
    CUSTOM_CATEGORIES.extend(value)

print(f"Using {len(CUSTOM_CATEGORIES)} custom categories for analysis.")

# Prompt 构建
def make_prompt(comment):
    return f"""
You are an expert in psychology and linguistic analysis.  
Given the following student graduation comment:

"{comment}"

Please rate the presence of each of the following dimensions on a scale from 0 to 1 (where 0 means not present at all, 1 means strongly present).  
Output JSON with format: {{'dimension_name': score, ...}}

Dimensions:
{', '.join(CUSTOM_CATEGORIES)}

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
            temperature=0
        )
        text = response.choices[0].message.content  # 新属性访问方式
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"JSON 解析失败，原始响应内容: {text}")
        return {dim: 0 for dim in CUSTOM_CATEGORIES}
    except Exception as e:
        print(f"[Error] {e}")
        return {dim: 0 for dim in CUSTOM_CATEGORIES}

# 🔄 处理批次
def process_batch(students, processed_names, batch_size=100, save_path="output_/output_dpsk/deepseek.csv"):
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
with open("all_data_use_labeled.json", "r", encoding="utf-8") as f:
    students = json.load(f)

# 📂 已完成记录
processed_names = set()
save_file = "output_/output_dpsk/deepseek.csv"
if os.path.exists(save_file):
    processed = pd.read_csv(save_file)
    processed_names = set(processed["name"].unique())

# 🔁 开始处理
process_batch(students, processed_names, batch_size=100, save_path=save_file)

# ✅ 完成
print("所有评论分析完成！结果已保存到", save_file)
