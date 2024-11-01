from tqdm import tqdm
import json
fin = open("/home/myjia/Medical_LLM_task/dataset/CMtMedQA.json", 'r', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/CMtMedQA_pifuke.json", 'w', encoding='utf-8')
fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/CMtMedQA_jingshenke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/CMtMedQA_zhongliuke.json", 'w', encoding='utf-8')

# 这个json文件里面是一个大的list，每个元素是一个dict，现在首先把这个list读取出来，然后按照每个元素依次处理
data = json.load(fin)
for i, line in tqdm(enumerate(data), total=len(data)):
    # line是一个dict，包含了query，response，label
    cate1 = line['cate1']
    cate2 = line['cate2']
    label = cate1 + cate2
    if "皮肤科" in label:
        fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in label or "心理内科" in label:
        fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in label:
        fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()