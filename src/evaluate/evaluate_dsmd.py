# coding=utf-8
# 计算f1值等指标
# 参考文献: AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis

import xlrd
from fuzzywuzzy import process
import json
from tqdm import tqdm
import os

class HyperParams():
    def __init__(self):
        self.num_dis = 2
        self.data_name = "GMD"
        self.fin_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/data/DSMD/{self.data_name}/output/qwen-7b-chat_dr_corom_emb_{self.num_dis}/result/"
        self.ref_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/data/DSMD/{self.data_name}/ref/"
        self.output_file = f"/home/myjia/Medical_LLM_task/EMR_diagnos/data/DSMD/{self.data_name}/output/qwen-7b-chat_dr_corom_emb_{self.num_dis}/result/result.json"
        self.ICD_database = '/home/myjia/Medical_LLM_task/EMR_diagnos/data/ICD/国际疾病分类ICD-10北京临床版v601.xls'
        self.top_n = 10
        self.threshold = 50

args = HyperParams()

database = args.ICD_database

xls = xlrd.open_workbook(database)
sheet = xls.sheet_by_index(0)
disease_ids = sheet.col_values(colx = 0, start_rowx = 1)
disease_names = sheet.col_values(colx = 1, start_rowx = 1)
ICD_disease = {}
for disease_id, disease_name in zip(disease_ids, disease_names): 
    ICD_disease[disease_name] = disease_id

fout = open(args.output_file, 'w', encoding="utf-8")

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

fin_pred = open(args.fin_dir + "cmd_pred_record.json", 'r', encoding="utf-8") if args.data_name == "CMD" else open(args.fin_dir + "gmd_pred_record.json", 'r', encoding="utf-8")
fin_ref = open(args.ref_dir + "ref.json", 'r', encoding="utf-8")

input_data_pred = []
for line in fin_pred:
    input_data_pred.append(json.loads(line))

input_data_ref = []
for line in fin_ref:
    input_data_ref.append(json.loads(line))

true_positive = 0.00001 # smooth
false_positive = 0
false_negitative = 0

for idx, EMR_dict in tqdm(enumerate(input_data_pred), total = len(input_data_pred), desc = f'开始进行评估'):
    line_ref = input_data_ref[idx]
    line_pred = EMR_dict
    assert line_ref["index"] == line_pred["index"]
    assert line_ref["index"] == idx + 1

    predictions = line_pred["pred_labels"]
    if len(predictions) == 0:
        predictions = ['*']
    
    ref_match = line_ref['ref_match']

    pred_match = [process.extract(r, ICD_disease.keys(), limit=args.top_n) for r in predictions]
    pred_match = [[(r[0], ICD_disease[r[0]], r[1]) for r in rr] for rr in pred_match]

    refs = [[n for n in m if n[2] >= args.threshold] for m in ref_match]
    preds = [[n for n in m if n[2] >= args.threshold] for m in pred_match]
    set_matched = [0] * len(refs)
    for pred in preds:
        set_match_idx = set_match(pred, refs, set_matched)
        if set_match_idx is None: false_positive += 1 # do not match
        elif set_matched[set_match_idx] == 1: false_positive += 1 # can not match more than one times
        else: set_matched[set_match_idx] = 1 # first match

    true_positive += sum(set_matched)
    false_negitative += (len(refs) - sum(set_matched))

set_recall = true_positive / (true_positive + false_negitative)
set_precision = true_positive / (true_positive + false_positive)
set_f1 = set_precision * set_recall * 2 / (set_recall + set_precision)

print(f"set_recall: {set_recall}")
print(f"set_precision: {set_precision}")
print(f"set_f1: {set_f1}")

fout.write(json.dumps({"set_recall": set_recall, "set_precision": set_precision, "set_f1": set_f1}, ensure_ascii=False) + "\n")
fout.close()