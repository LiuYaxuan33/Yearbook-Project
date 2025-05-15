from paddleocr import PaddleOCR
import requests
import json
import os
from datetime import datetime

# 初始化 OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",          # 指定英文语言
    det_model_dir='en_PP-OCRv3_det',  # 英文检测模型
    rec_model_dir='en_PP-OCRv3_rec',  # 英文识别模型
    cls_model_dir='ch_ppocr_mobile_v2.0_cls'  # 保持中文方向分类器
)

# DeepSeek API配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-e5faf4be216d4396b8db95c40d88f574"  # 替换为你的实际API密钥
MODEL_NAME = "deepseek-chat"

def process_image_folder(folder_path, output_file):
    # 获取支持的图片格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    # 遍历文件夹并过滤图片文件
    image_files = sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        print("未找到支持的图片文件（支持格式：PNG/JPG/JPEG/BMP/TIFF/WEBP）")
        return

    print(f"发现 {len(image_files)} 张待处理图片，开始处理...")

    # 创建结果保存文件
    with open(output_file, "w", encoding="utf-8") as result_file:
        result_file.write(f"OCR处理结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"正在处理 ({idx}/{len(image_files)}): {os.path.basename(image_path)}")
                
                # OCR处理
                ocr_text = ocr_with_paddle(image_path)
                
                # 文本润色
                rewritten_text = deepseek_rewrite(ocr_text)
                
                # 写入结果
                result_file.write(f"图片名称: {os.path.basename(image_path)}\n")
                result_file.write(f"处理结果:\n{rewritten_text}\n")
                result_file.write("-" * 50 + "\n\n")
                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {str(e)}")
                result_file.write(f"图片名称: {os.path.basename(image_path)} [处理失败]\n\n")

    print(f"\n处理完成！结果已保存至：{os.path.abspath(output_file)}")

def deepseek_rewrite(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                 "content": 
"""
Perform text refinement with these priorities:
1. Fundamental Corrections:
   - Capitalize proper nouns (e.g.: john doe → John Doe, ARNOLD → Arnold)
   - Fix letter confusions (e.g.: cl → d, rn → m, 1 → l， 0 → o)
   - Correct obvious spelling errors (e.g.: hel1o → hello)
   - Complete missing initials of common surnames (e.g.: pple → Apple, Arry → Barry)

2. Paragraph Optimization:
   - Merge erroneous line breaks (e.g.: "Please submit\nthe report" → "Please submit the report")
   - Preserve intentional paragraph breaks (e.g.: section headings)
   - Reorganize paragraphs based on semantic coherence

3. Output Requirements:
   - Maintain original information integrity
   - Return only the polished text
   - Use standard English punctuation

Example:
[Raw OCR]
Project proqress Report:
As of this week, Team A has completed
80% of the tasl. The main chal1enge
is data co1lection delays. Team B
progress is s1ower than expected.

[Refined]
Project Progress Report:
As of this week, Team A has completed 80% of the task. 
The main challenge is data collection delays. 
Team B progress is slower than expected.

Provide only the polished version
"""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API请求失败，返回原始文本。错误信息: {str(e)}")
        return text

def ocr_with_paddle(image_path):
    result = ocr.ocr(image_path, cls=True)
    return "\n".join([word_info[1][0] for line in result for word_info in line])

# 使用示例
if __name__ == "__main__":
    # 设置包含图片的文件夹路径
    image_folder = "/Users/liuyaxuan/Desktop/25春/25春/RA_YilingZhao/refined_yearbook/1911"
    
    # 设置输出文件名（可选）
    output_filename = "/Users/liuyaxuan/Desktop/25春/25春/RA_YilingZhao/refined_yearbook/1911.txt"
    
    process_image_folder(image_folder, output_filename)