import json

with open('all_data_use_labeled.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for person in data:
    if person['is_agriculture'] == 1:
        if person['gender'] == 'Female': 
            print(f"{person['name']}")