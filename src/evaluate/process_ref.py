# coding=utf-8
# 计算f1值等指标
# 参考文献: AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis

import xlrd
from fuzzywuzzy import process
import json
from tqdm import tqdm

class HyperParams():
    def __init__(self):
        self.input_file = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/DSMD/GMD/gmd.json"
        self.ICD_database = '/home/myjia/Medical_LLM_task/EMR_diagnos/data/ICD/国际疾病分类ICD-10北京临床版v601.xls'
        self.top_n = 10
        self.threshold = 50
        # self.output_log = "/home/myjia/Medical_LLM_task/EMR_diagnos/output/result/result_record_baseline_EMR.xlsx"

args = HyperParams()

database = args.ICD_database

xls = xlrd.open_workbook(database)
sheet = xls.sheet_by_index(0)
disease_ids = sheet.col_values(colx = 0, start_rowx = 1)
disease_names = sheet.col_values(colx = 1, start_rowx = 1)
ICD_disease = {}
for disease_id, disease_name in zip(disease_ids, disease_names): 
    ICD_disease[disease_name] = disease_id


def set_match(pred, refs, matched):
    pred_set = [p[0] for p in pred]
    return_idx = None
    for idx, ref in enumerate(refs):
        ref_set = [r[0] for r in ref]
        for p in pred_set:
            for r in ref_set:
                if p == r and matched[idx] == 0:
                    return_idx = idx
    return return_idx

true_positive = 0.00001 # smooth
false_positive = 0
false_negitative = 0

# 按行读取json文件

fin = open(args.input_file, 'r', encoding='utf-8')

input_data = json.load(fin)


output_record = []

for idx, line in tqdm(enumerate(input_data), total=len(input_data)):
    # line = json.loads(line)
    

    references = line['label']
    # predictions = line_pred['pred_response']
    index = line['index']

    assert index == idx + 1


    ref_match = [process.extract(r, ICD_disease.keys(), limit=args.top_n) for r in references] # 模糊查询过程，得到的是一个列表，列表中的每个元素是一个元组，元组中的第一个元素是匹配到的icd列表里的疾病，第二个元素是匹配的分数，比如[('高血压', 100), ('1型糖尿病性高血压', 90), ('1型糖尿病性肥胖症性高血压', 90), ('2型糖尿病性高血压', 90), ('2型糖尿病性肥胖症性高血压', 90), ('糖尿病性高血压', 90), ('糖尿病性肥胖症性高血压', 90), ('糖耐量受损伴肥胖型高血压', 90), ('糖耐量受损伴高血压', 90), ('高血压所致精神障碍', 90)]
    ref_match = [[(r[0], ICD_disease[r[0]], r[1]) for r in rr] for rr in ref_match]

    # pred_match = [process.extract(r, ICD_disease.keys(), limit=args.top_n) for r in predictions]
    # pred_match = [[(r[0], ICD_disease[r[0]], r[1]) for r in rr] for rr in pred_match]

    

    output_record.append({'index':index, 'ref_match':ref_match, 'ref_response':references})


# 保存结果


fout = open("/home/myjia/Medical_LLM_task/EMR_diagnos/data/DSMD/GMD/ref/ref.json", 'w', encoding='utf-8')

for record in output_record:
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.flush()

