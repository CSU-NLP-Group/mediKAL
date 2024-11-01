from tqdm import tqdm
# 获取2010-2020年的数据
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/Medical-Dialogue-Dataset-Chinese/Medical-Dialogue-Dataset-Chinese/MDDC_pifuke.txt", 'w', encoding='utf-8')
fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/Medical-Dialogue-Dataset-Chinese/Medical-Dialogue-Dataset-Chinese/MDDC_jingshenke.txt", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/Medical-Dialogue-Dataset-Chinese/Medical-Dialogue-Dataset-Chinese/MDDC_zhongliuke.txt", 'w', encoding='utf-8')
for i in tqdm(range(2010, 2021)):
    fin = open("/home/myjia/Medical_LLM_task/dataset/Medical-Dialogue-Dataset-Chinese/Medical-Dialogue-Dataset-Chinese/{}.txt".format(i), 'r', encoding='utf-8')
    sample = []
    flag = 0
    next_faculty = 0
    for i,line in tqdm(enumerate(fin)):
        if line.startswith("id="):
            if len(sample):
                if flag == 1:
                    for item in sample:
                        fout_pifuke.write(item)
                        fout_pifuke.flush()
                elif flag == 2:
                    for item in sample:
                        fout_jingshenke.write(item)
                        fout_jingshenke.flush()
                elif flag == 3:
                    for item in sample:
                        fout_zhongliuke.write(item)
                        fout_zhongliuke.flush()
                # else:
                #     continue
                sample = []
        elif line.startswith("Doctor faculty"):
            next_faculty = 1
            # continue
        elif next_faculty == 1:
            if "皮肤科" in line:
                flag = 1
            elif "精神科" in line:
                flag = 2
            elif "肿瘤科" in line:
                flag = 3
            else:
                flag = 0
            next_faculty = 0
            # continue
        sample.append(line)
    fin.close()    