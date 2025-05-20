import json

with open('all_data_use_labeled.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


ALL_LABELS = [
    "is_agriculture",
    "is_home economics",
    "is_science",
    "is_engineering",
    "is_music",
    "is_education",
    "is_veterinary"]

MAJOR_TO_ANALYZE = ALL_LABELS[0] 

for labels in ALL_LABELS:
    man = 0
    woman = 0
    for person in data:
        if person[labels] == 1:
            if person['gender'] == 'Male':      
                man += 1
            else:  
                woman += 1
    print(f"{labels}: man has {man} and woman has {woman}")


