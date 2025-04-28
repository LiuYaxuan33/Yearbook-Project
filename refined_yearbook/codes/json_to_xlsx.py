import json
import pandas as pd

# 定义需要检测的俱乐部列表
CLUBS_CHECKLIST = {
    'if_in_Home_Economics_Club': 'Home Economics Club',
    'if_in_Iowa_Homemaker': 'Homemaker',
    'if_in_Home_Economics_Council': 'Home Economics Council',
    'if_in_Omicron_Nu': 'Omicron Nu'
}

def process_entry(entry):
    """处理单个JSON条目"""
    #print(type(entry))
    processed = {
        'name': entry['name'],
        'gender': 0 if entry['gender'].lower() == 'female' else 1,
        'major': entry['major'],
        'hometown': entry['hometown'],
        'clubs': '; '.join(entry['clubs'])
    }
    
    # 添加俱乐部检测字段
    for field, club_name in CLUBS_CHECKLIST.items():
        processed[field] = 1 if club_name in '; '.join(entry['clubs']) else 0
    
    return processed

# 读取JSON文件
with open('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/refined_yearbook/results, 22-32/1932.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理所有条目
processed_data = [process_entry(entry) for entry in data]

# 创建DataFrame并保存为Excel
df = pd.DataFrame(processed_data)
df.to_excel('1932.xlsx', index=False, columns=[
    'name',
    'gender',
    'major',
    'hometown',
    'clubs',
    *CLUBS_CHECKLIST.keys()  # 展开所有检测字段
])