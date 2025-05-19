
from paddleocr import PaddleOCR
import requests
from openai import OpenAI
import json
import base64
import re
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  # 加载.env文件中的环境变量


# 初始化 OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    det_model_dir='en_PP-OCRv3_det',
    rec_model_dir='en_PP-OCRv3_rec',
    cls_model_dir='ch_ppocr_mobile_v2.0_cls'
)

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
MODEL_NAME = "deepseek-chat"

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ocr_with_paddle(image_path):
    result = ocr.ocr(image_path, cls=True)
    return "\n".join([word_info[1][0] for line in result for word_info in line])

def ocr_with_qwen(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    img_format = image_path.split('.')[-1].lower()
    mime_type = f"image/{img_format}" if img_format in ['png', 'jpeg', 'jpg', 'webp'] else "image/jpeg"

    completion = client.chat.completions.create(
        model="qwen-vl-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                    {"type": "text", "text": 
"""
请严格按照以下要求执行OCR识别：
1. 全面扫描：
   - 按从左到右、从上到下的顺序逐行扫描
   - 确保每行都完整识别从最左端到最右端的内容
   - 特别注意行末可能被截断的单词
   - 特别注意靠右对齐的零散内容如“Ames, Ia.”或“C.E.”
   - 自上而下逐行识别，在同一行的内容（不管中间有多大空白）全部识别完成后再识别下一行。

2. 布局保留：
   - 严格保持原始换行符（包括空行）
   - 行内连续文本不要擅自添加换行
   - 保留段落之间的自然空行

3. 特殊处理：
   - 连字符结尾的行（如"electro-")要与下一行衔接
   - 识别表格/分栏内容时，按视觉顺序而非逻辑顺序
   - 数字0和字母O要结合上下文区分

4. 完整性检查：
   - 如果行末字符靠近图片边缘，需二次确认
   - 对模糊区域采用概率采样而非直接跳过
"""
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content.strip()

def process_image_folder(folder_path, output_file):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        print("未找到支持的图片文件")
        return

    all_students = []
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 新增合并文件路径
    merged_ocr_path = os.path.join(output_dir, "merged_ocr_results.txt")
    merged_rewritten_path = os.path.join(output_dir, "merged_rewritten_results.txt")
    

    with open(merged_ocr_path, "w", encoding="utf-8") as ocr_merged_file, \
         open(merged_rewritten_path, "w", encoding="utf-8") as rewritten_merged_file:

        for idx, image_path in enumerate(image_files, 1):
            try:
                image_name = os.path.basename(image_path)
                print(f"正在处理 ({idx}/{len(image_files)}): {image_name}")

                # OCR处理
                ocr_text_1 = ocr_with_qwen(image_path)
                ocr_text_2 = ocr_with_paddle(image_path)
                ocr_text = "A:\n"+ocr_text_1 + "===\n" + "B:\n"+ ocr_text_2
                
                # 写入合并OCR文件
                ocr_merged_file.write(f"=== {image_name} ===\n{ocr_text}\n\n")

                # 文本润色
                rewritten_text = deepseek_rewrite(ocr_text)
                
                # 写入合并润色文件
                rewritten_merged_file.write(f"=== {image_name} ===\n{rewritten_text}\n\n")

                # 格式转换
                student_json = deepseek_to_json(rewritten_text, output_file)
                
                if student_json:
                    if type(student_json) == list:
                        all_students.extend(student_json) 
                    else:
                        if type(student_json) == dict:
                            all_students.append(student_json)

                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {str(e)}")

    # 写入JSON结果
    if all_students:
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(all_students, json_file, ensure_ascii=False, indent=2)
        print(f"\n处理完成！生成文件："
              f"\n- 最终JSON：{os.path.abspath(output_file)}"
              f"\n- 合并OCR结果：{os.path.abspath(merged_ocr_path)}"
              f"\n- 合并润色结果：{os.path.abspath(merged_rewritten_path)}")
        
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
Perform OCR fusion and text refinement with these priorities:

Dual-Input Processing:
Cross-check two OCR inputs (Input A: accurate punctuation but potential text omissions; Input B: complete text but missing punctuation)
Merge texts using B's content as base while preserving A's punctuation
Recover missing text fragments through character-level alignment
Comprehensive Corrections:
Resolve character confusions (rn→m, cl→d, 1/i/l differentiation)
Complete partial words (pple→Apple, micr→micro)
Restore missing punctuation:
Sentence-ending marks (./!/?)
Quotation marks and hyphens
List formatting (commas/semicolons)
Fix line-break errors while preserving valid paragraph breaks
Structural Integrity:
Maintain original section breaks (detect headers/transitions through formatting patterns)
Insert paragraph spacing between distinct information blocks (e.g., student profiles)
Remove pagination artifacts (page numbers/headers/footers)
Output Requirements:
Return only the enhanced text
Preserve all informational elements
Use standard English formatting
Never add explanatory comments
Processing Example:
[Input A] Customer Name: "John_Doe";
Addr: 123 Oak St...

[Input B] Customer Name John Doe
Addr 123 Oak St

[Output] Customer Name: "John Doe";
Address: 123 Oak St...

Provide only the pure polished text without any markdown or additional explanations.
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

def deepseek_to_json(text, output_file):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": """
请严格按以下规则解析学生信息为JSON格式：
1. 字段顺序必须为：name, gender, hometown, nickname, major, clubs, comment
2. 姓名格式规则：
   - 检查姓氏，如果和常见姓氏相比缺失首字母（如Barry -> Arry）则补全
   - 每个单词的首字母大写，其余字母小写
   - 姓氏在前，名在后，二者以逗号隔开（即常见的排版方式）
3. 性别推断基于英文名常见性别（Male/Female）
4. 籍贯识别：
   - 匹配已知国家/城市名格式（如New York, London）
5. 昵称识别：
   - 常在双引号中
   - 保留原有昵称
   - 将所有昵称（如有）以列表形式储存
   - 零散的单词或短语；双引号引起的俚语、短句不包含在内
   - 通常较短
6. 专业识别：
   - 保留原有专业名称或缩写（如H. Ec., Vet., C. E., Agr.等）
7. 社团信息：
   - 不成句的单词/词组
   - 每行一个条目
   - 没有则留空列表
   - 社团与社团之间常用连字符'-'隔开
   - 包含可能的college或university信息
8. 评语：
   - 完整的句子
   - 去除原始换行符
   - 保留原有标点符号，如有需要，使用转义符
   - 包含双引号引起的俚语、短句，注意与昵称区分
   - 如果评语内部出现双引号"，记得使用转义符以避免格式错误
9. 保留原始信息，除非明显错误

示例输入：
LBECHT, L.R. Tama, Iowa  
“dad” C.E.
Pi Beta Phi-Philomathean

"When joy and duty clash, let duty go to smash."

Ames found a loyal convert in Louise after sojourns at Leander Clark and Cornell. A great student at times. Prefers Domestic Science because of its excellent home training. A girl with whom to have the best kind of a time.  

示例输出：
{
  "name": "Albrecht, L. R.",
  "gender": "Male",
  "hometown": "Tama, Iowa",
  "nicknames": ["dad"],
  "major": "C.E.",
  "clubs": ["Pi Beta Phi","Philomathean"],
  "comment": "\"When joy and duty clash, let duty go to smash.\" Ames found a loyal convert in Louise after sojourns at Leander Clark and Cornell. A great student at times. Prefers Domestic Science because of its excellent home training. A girl with whom to have the best kind of a time."
}

只需返回JSON，不要添加任何代码块标记（如markdown格式的```json），只返回纯文本，也不要额外解释。把所有学生信息放在同一个列表中。

请特别注意：当评语内容包含双引号时，必须使用反斜杠进行转义。例如：\"She said \\\"hello\\\"\"。生成的JSON必须直接可以被Python的json.loads()解析。
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
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        json_str = response.json()['choices'][0]['message']['content'].strip()
        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON转换失败: {str(e)}")
        print(json_str)
        return None

# 使用示例
if __name__ == "__main__":
    image_folder = "/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/pics/1906-double/6"
    output_file = "/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1906-6.json"
    
    process_image_folder(image_folder, output_file)