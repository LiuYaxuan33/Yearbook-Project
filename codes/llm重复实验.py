import openai
import pandas as pd
import json
import os
import re
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

# ✅ 初始化客户端（新版本API修改点1）
client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"  # DeepSeek专属端点
)

with open("dimensions.json", "r", encoding="utf-8") as f0:
    dimension_all = json.load(f0)

framework = dimension_all[0]
root_words = dimension_all[1]

CUSTOM_CATEGORIES = []

for key, value in framework.items():
    CUSTOM_CATEGORIES.extend(value)

print(CUSTOM_CATEGORIES)

# 分析维度

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



# 📂 已完成记录
processed_names = set()
save_file = "deepseek_emotion_scores.csv"
if os.path.exists(save_file):
    processed = pd.read_csv(save_file)
    processed_names = set(processed["name"].unique())


# 稳定性测试参数配置
STABILITY_CONFIG = {
    "num_samples": 100,      # 随机抽取样本数量
    "num_runs": 10,        # 每个样本重复次数
    "results_file": "0520_stability_results.csv",  # 结果存储文件
    "samples_file": "0520_stability_samples.json", # 抽样记录文件
    "sleep_time": 1.2       # API调用间隔
}

def run_stability_test():
    """执行稳定性测试主函数"""
    # 加载所有学生数据
    with open("all_data_use_labeled.json", "r", encoding="utf-8") as f:
        all_students = json.load(f)
    
    # 获取或创建抽样样本
    selected_students = _get_samples(all_students)
    
    # 加载已处理记录
    processed = _load_processed_records()
    
    # 计算剩余任务量
    total_tasks = _calculate_remaining_tasks(selected_students, processed)
    
    # 执行测试任务
    with tqdm(total=total_tasks, desc="Stability Test Progress") as pbar:
        for student in selected_students:
            name = student["name"]
            remaining_runs = _get_remaining_runs(name, processed)
            
            for run_id in remaining_runs:
                _process_single_run(student, run_id)
                _update_progress(pbar, name, run_id, processed)
                
                time.sleep(STABILITY_CONFIG["sleep_time"])
    
    # 结果分析与可视化
    analyze_stability_results()

def _get_samples(all_students):
    """获取或创建抽样样本"""
    if os.path.exists(STABILITY_CONFIG["samples_file"]):
        with open(STABILITY_CONFIG["samples_file"], "r", encoding='utf-8') as f:
            return json.load(f)
    
    selected = random.sample(all_students, STABILITY_CONFIG["num_samples"])
    with open(STABILITY_CONFIG["samples_file"], "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    return selected

def _load_processed_records():
    """加载已处理记录"""
    processed = defaultdict(set)
    if os.path.exists(STABILITY_CONFIG["results_file"]):
        df = pd.read_csv(STABILITY_CONFIG["results_file"])
        for _, row in df.iterrows():
            processed[row["name"]].add(row["run_id"])
    return processed

def _calculate_remaining_tasks(students, processed):
    """计算剩余任务总数"""
    return sum(
        STABILITY_CONFIG["num_runs"] - len(processed.get(s["name"], set()))
        for s in students
    )

def _get_remaining_runs(name, processed):
    """获取需要运行的剩余run_id"""
    completed = processed.get(name, set())
    return [
        run_id for run_id in range(1, STABILITY_CONFIG["num_runs"] + 1)
        if run_id not in completed
    ]

def _process_single_run(student, run_id):
    """处理单个运行实例"""
    result = analyze_comment(student["comment"])
    record = {
        "name": student["name"],
        "gender": student.get("gender", ""),
        "year": student.get("year", ""),
        "run_id": run_id,
        **result
    }
    pd.DataFrame([record]).to_csv(
        STABILITY_CONFIG["results_file"],
        mode="a",
        header=not os.path.exists(STABILITY_CONFIG["results_file"]),
        index=False
    )

def _update_progress(pbar, name, run_id, processed):
    """更新进度和记录"""
    pbar.update(1)
    processed[name].add(run_id)

def analyze_stability_results():
    """分析结果并生成可视化"""
    df = pd.read_csv(STABILITY_CONFIG["results_file"])
    
    # 计算统计指标
    stats = df.groupby("name")[CUSTOM_CATEGORIES].agg(["mean", "std", "sem"])
    
    # 维度稳定性分析
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[CUSTOM_CATEGORIES])
    plt.title("Score Distribution Across All Runs")
    plt.xticks(rotation=45)
    plt.show()
    
    # 各维度变异系数分析
    cv = stats.xs("std", axis=1, level=1).mean() / stats.xs("mean", axis=1, level=1).mean()
    cv.plot(kind="bar", title="Coefficient of Variation by Dimension")
    plt.ylabel("CV (std/mean)")
    plt.show()
    
    # 样本稳定性热力图
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        stats.xs("std", axis=1, level=1).T,
        annot=False,
        cmap="YlGnBu",
        cbar=True
    )
    plt.title("Standard Deviation Heatmap (Columns: Samples, Rows: Dimensions)")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

    heatmap_values = stats.xs("std", axis=1, level=1).T

    heatmap_values.to_csv("heatmap_values.csv")


# 在原有代码后添加调用（确保原有流程被注释）
# process_batch(students, processed_names, ...)  # 注释原有处理流程
run_stability_test()  # 执行稳定性测试