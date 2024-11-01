import json
from tqdm import tqdm

fin = open("/home/myjia/Medical_LLM_task/dataset/firefly_chatglm3/shuffled_data_train.json", "r", encoding="utf-8")
fout = open("/home/myjia/Medical_LLM_task/LLMs_hub/LLaMA-Factory/data/pifuke/train.json", "w", encoding="utf-8")


total_data = []
for i, line in tqdm(enumerate(fin),total=178316):
    data = json.loads(line)
    conversations = data["conversations"]
    new_conv = {}
    pair_flag = 0
    all_history = []
    history = []
    for j,conv in enumerate(conversations):
        new_conv['input'] = ""
        content = conv["content"]
        if pair_flag == 0:
            new_conv["instruction"] = content
            history.append(content)
            pair_flag = 1
        else:
            new_conv["output"] = content
            history.append(content)
            all_history.append(history)
            if j == 1:
                new_conv['history'] = []
            else:
                new_conv["history"] = all_history
            pair_flag = 0
            total_data.append(new_conv)
            new_conv = {}
            history = []

# 将total_data这个列表整体写入json文件
json.dump(total_data, fout, ensure_ascii=False)

