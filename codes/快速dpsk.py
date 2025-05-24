import asyncio
import httpx
import pandas as pd
import json
import os
import re
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# 自定义维度加载
with open("dimensions.json", "r", encoding="utf-8") as f:
    dimension_all = json.load(f)
framework = dimension_all[0]
CUSTOM_CATEGORIES = [v for values in framework.values() for v in values]

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

# 🔄 发送请求
async def analyze_comment_async(client, comment):
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": make_prompt(comment)}],
            "temperature": 0
        }

        resp = await client.post(BASE_URL, json=json_data, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)

    except Exception as e:
        print(f"[Error] {e}")
        return {dim: 0 for dim in CUSTOM_CATEGORIES}

# 批处理评论
async def process_batch_async(students, processed_names, save_path="output_/output_dpsk/deepseek.csv", batch_size=20):
    results = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(students), batch_size):
            batch = students[i:i+batch_size]
            tasks = []

            for student in batch:
                if student["name"] in processed_names:
                    continue
                tasks.append(analyze_comment_async(client, student["comment"]))

            responses = await tqdm_asyncio.gather(*tasks, desc=f"Processing batch {i//batch_size+1}")

            for j, result in enumerate(responses):
                student = batch[j]
                result.update({
                    "name": student["name"],
                    "gender": student.get("gender", ""),
                    "year": student.get("year", "")
                })
                results.append(result)
                processed_names.add(student["name"])

            # 每批写入
            if results:
                df = pd.DataFrame(results)
                df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))
                results.clear()

# 启动主流程
async def main():
    with open("all_data_use_labeled.json", "r", encoding="utf-8") as f:
        students = json.load(f)

    processed_names = set()
    save_file = "output_/output_dpsk/deepseek.csv"
    if os.path.exists(save_file):
        processed = pd.read_csv(save_file)
        processed_names = set(processed["name"].unique())

    await process_batch_async(students, processed_names, save_path=save_file)

    print("✅ 所有评论分析完成！")

# 运行异步主函数
asyncio.run(main())
