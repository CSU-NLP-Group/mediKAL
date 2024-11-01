import json

fin_ref = open("/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/processed_json_files/ref/内科_ref.json", "r", encoding="utf-8")

fin_pred = open("/home/myjia/Medical_LLM_task/EMR_diagnos/output/result/qwen-72b-chat/内科_res.json", "r", encoding="utf-8")

refs_data = []
for line in fin_ref:
    refs_data.append(json.loads(line))

preds_data = []
for line in fin_pred:
    preds_data.append(json.loads(line))


def overlap_percentage(str1, str2, threshold=0.5):
    """Check if two strings overlap based on character similarity percentage."""
    len1 = len(str1)
    len2 = len(str2)
    
    set1 = set(str1)
    set2 = set(str2)
    common_chars = set1.intersection(set2)

    # Calculate overlap percentage
    overlap_percent = len(common_chars) / max(len1, len2)
    
    # Check if overlap percentage meets the threshold
    if overlap_percent >= threshold:
        return True
    else:
        return False

def check(str1, list1):
    """Check if a string is in a list of strings."""
    for str2 in list1:
        if overlap_percentage(str1, str2):
            return True
    return False

useful = 0
cross = 0

for i in range(3000):
    ref = refs_data[i]
    pred = preds_data[i]
    assert ref["index"] == pred["index"]

    pred_labels = pred["pred_labels"]

    ref_match = ref["ref_match"]

    ref_list = []

    for ref in ref_match:
        for r in ref:
            ref_list.append(r[0])

    original_pred = pred["pred_response"][-1][0]

    for orig_pre in original_pred:
        if check(str1=orig_pre, list1=ref_list):
            useful += 1
            if orig_pre in pred_labels:
                cross += 1

print(useful, cross, cross / useful)
            
