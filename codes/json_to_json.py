import json

# 读取原始JSON文件（处理可能的编码和结构问题）
try:
    with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/1922.json', 'r', encoding='utf-8') as f:
        data = json.load(f)  # 确认读取的是列表结构
    
    # 添加类型安全检查
    if not isinstance(data, list):
        raise ValueError("JSON文件根元素不是列表，请检查文件结构")

    # 筛选逻辑增强
    female_data = [
        item for item in data 
        if isinstance(item, dict) and  # 确保是字典
        str(item.get('gender', '')).lower() == 'female'  # 处理大小写和空值
    ]

    # 写入新文件
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(female_data, f, ensure_ascii=False, indent=4)  # 保留非ASCII字符

    print(f"成功提取 {len(female_data)} 条记录，已保存到output.json")

except Exception as e:
    print(f"处理失败，错误类型：{type(e).__name__}")
    print(f"错误详情：{str(e)}")