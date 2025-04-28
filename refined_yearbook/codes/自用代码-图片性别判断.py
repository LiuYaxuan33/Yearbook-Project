from openai import OpenAI
import json
import base64
import os
import re
import time

QWEN_API_KEY = 'sk-2b1d01e7d476419bb27fa43f3fe17d22'

# 初始化OpenAI客户端（请替换为您的实际API信息）
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def detect_gender(image_path):
    """使用Qwen模型检测图片中人物的性别"""
    try:
        # 读取并编码图片
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # 确定MIME类型
        img_format = image_path.split('.')[-1].lower()
        mime_type = f"image/{img_format}" if img_format in ['png', 'jpeg', 'jpg', 'webp'] else "image/jpeg"

        # 调用API
        completion = client.chat.completions.create(
            model="qwen-omni-turbo",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Identify the gender of each person in the image. "
                                "There are 4 people in each image. "
                                "Output in the format:\n"
                                "Person 1: Male/Female/Uncertain\n"
                                "Person 2: Male/Female/Uncertain"
                            )
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                        {"type": "text", "text": "Identify the gender."}
                    ],
                },
            ],
            modalities=["text"],
            stream=True
        )

        # 处理流式响应
        result = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content

        # 解析结果
        genders = []
        for line in result.strip().split("\n"):
            if "Person" in line and ":" in line:
                gender = line.split(":")[1].strip()
                if gender in ["Male", "Female"]:
                    genders.append(gender)
                else:
                    genders.append("Uncertain")
        return genders
    
    except Exception as e:
        raise Exception(f"API调用失败: {str(e)}")

def process_images(folder_path):
    """处理指定文件夹中的所有图片"""
    # 获取并排序图片文件
    valid_ext = ('png', 'jpg', 'jpeg', 'webp')
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    ], key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    if not image_files:
        print("⚠️ 未找到图片文件")
        return

    results = []
    total = len(image_files)
    start_time = time.time()

    for idx, filename in enumerate(image_files, 1):
        file_path = os.path.join(folder_path, filename)
        result = {
            "filename": filename,
            "status": "success",
            "processing_time": 0,
            "genders": []
        }

        try:
            print(f"\n🚀 正在处理 [{idx}/{total}] {filename}")
            single_start = time.time()
            
            # 执行性别检测
            genders = detect_gender(file_path)
            result["genders"] = genders
            result["processing_time"] = time.time() - single_start
            
            print(f"✅ 完成！检测到 {len(genders)} 人：{genders}")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"❌ 处理失败：{str(e)}")

        results.append(result)

    # 保存结果
    output_path = os.path.join(folder_path, "1913_gender_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    total_time = time.time() - start_time
    success = sum(1 for r in results if r["status"] == "success")
    print("\n" + "="*50)
    print(f"📊 处理完成！共 {total} 张图片")
    print(f"✅ 成功: {success} | ❌ 失败: {total - success}")
    print(f"⏱️ 总耗时: {total_time:.2f}s | 平均: {total_time/total:.2f}s/张")
    print(f"💾 结果已保存至: {output_path}")

if __name__ == "__main__":
    # 使用示例
    target_folder = "/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1913"
    process_images(target_folder)