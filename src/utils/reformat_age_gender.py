import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import openpyxl
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from neo4j import GraphDatabase, basic_auth
from graphdatascience import GraphDataScience
from retriv import SparseRetriever

# import zhipuai
# from langchain.llms import ChatGLM
from tqdm import tqdm
import numpy as np
import pandas as pd

import time
import sys
import re
import math

class hyperparams:
    def __init__(self):
        # self.khops = 1

        # 数据相关 
        self.dataset_name = "EMR_excel"
        self.dataset_path = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/{}.xlsx".format(self.dataset_name)

        # 大模型相关
        self.model_version = "baichuan2-7b-chat" # [baichuan2-7b-chat, chatglm3-6b]
        self.model_name_or_path = "/home/myjia/Medical_LLM_task/LLMs_base_model/{}".format(self.model_version)
        self.model_type = "baichuan"

        # NER模型
        self.ner_model_id = "/home/myjia/Medical_LLM_task/PLMs_base_model/nlp_raner_named-entity-recognition_chinese-base-cmeee"

        # retriever
        self.retriever_version = "bm25_merge_kg"

        # 输出相关
        self.output_path = "/home/myjia/Medical_LLM_task/EMR_diagnos/output/result/result_record_EMR.xlsx"
        # self.result_log_ref = "/home/myjia/Medical_LLM_task/EMR_diagnos/output/ref/log_baseline_EMR_ref.json"
        self.result_log_pred = "/home/myjia/Medical_LLM_task/EMR_diagnos/output/pred/log_KG_sum.json"

        # neo4j，知识图谱相关参数
        self.uri = "bolt://localhost:7687"
        self.username = "neo4j"
        self.password = "123456"
        self.EMR2KG_entity_path = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/EMR_ee_match/EMR2KG_entity.json"
        self.kg_type = "merge"
        self.kg_entity_path = '/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_{}.txt'.format(self.kg_type)
        self.entity_type_map_path = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/entity_type_maps/entity_type_map_{}.json".format(self.kg_type)
        self.khops = 5

args = hyperparams()

def load_model(model_type, model_name_or_path):
    """ 加载模型 """
    if model_type == "glm":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        chat_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        # if args.model_version == "-6b":
        #     chat_model = chat_model.half().cuda()
        chat_model = chat_model.cuda()
        chat_model.eval()
    elif model_type == "baichuan":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
        chat_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        Generation_Config = GenerationConfig.from_pretrained(model_name_or_path)
        Generation_Config.do_sample = False
        Generation_Config.temperature = None
        Generation_Config.top_p = None
        Generation_Config.top_k = None
        # Generation_Config.temperature = 0.01
        chat_model.generation_config = Generation_Config
        chat_model = chat_model.cuda().eval()
    # baichuan-13b量化加载
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    # chat_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # chat_model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
    # chat_model.quantize(8).cuda()
    
    return tokenizer, chat_model

def chat_glm(messages):
    """ glm调用chat """
    # if len(prompt) > 2048:
    #     prompt = prompt[:2048]
    if len(messages) == 1:
        prompt = messages[0]
        old_history = None
    else:
        prompt = messages[0]
        old_history = messages[1]

    if old_history is not None:
        response, history = chat_model.chat(tokenizer,
                                       prompt,
                                       do_sample=False,
                                       temperature=1.0,
                                       top_p = 1.0,
                                       repetition_penalty = 1.1,
                                       history = old_history
                                       )
    else:
        response, history = chat_model.chat(tokenizer,
                                       prompt,
                                       do_sample=False,
                                       temperature=1.0,
                                       top_p = 1.0,
                                       repetition_penalty = 1.1
                                       )
    return response, history

def chat_baichuan(messages):
    """ baichuan调用chat """
    # prompt = texts[0]
    # messages = []
    # messages.append({"role": "user", "content": prompt})
    response = chat_model.chat(tokenizer, messages)
    return response, []

def chat_(model_type, messages):
    """ 自定义chat接口 """
    if model_type == "glm":
        return chat_glm(messages)
    elif model_type == "baichuan":
        return chat_baichuan(messages)
    # elif model_type == "pro":
    #     return chat_pro(prompt)
    else:
        raise ValueError("model_type must be in ['glm', 'baichuan', 'pro']")
    
def get_disease_info(disease_name):
    # 按照关系类型查询实体的邻居实体
    gender_list = []
    pop_list = []
    age_list = []
    bod_list = []

    query1 = """
    MATCH (e)-[r:`发病性别倾向`]-(n)
    WHERE e.name = $disease_name
    RETURN collect(n.name) AS neighbor_entities
    """
    result1 = session.run(query1, disease_name=disease_name)
    for record in result1:
        neighbors1 = record["neighbor_entities"]
        gender_list.extend(neighbors1)

    query2 = """
    MATCH (e)-[r:`多发群体`]-(n)
    WHERE e.name = $disease_name
    RETURN collect(n.name) AS neighbor_entities
    """
    result2 = session.run(query2, disease_name=disease_name)
    for record in result2:
        neighbors2 = record["neighbor_entities"]
        pop_list.extend(neighbors2)
    
    query3 = """
    MATCH (e)-[r:`发病年龄`]-(n)
    WHERE e.name = $disease_name
    RETURN collect(n.name) AS neighbor_entities
    """
    result3 = session.run(query3, disease_name=disease_name)
    for record in result3:
        neighbors3 = record["neighbor_entities"]
        age_list.extend(neighbors3)
    
    # query4 = """
    # MATCH (e)-[r:`发病部位`]-(n)
    # WHERE e.name = $disease_name
    # RETURN collect(n.name) AS neighbor_entities
    # """
    # result4 = session.run(query4, disease_name=disease_name)
    # for record in result4:
    #     neighbors4 = record["neighbor_entities"]
    #     bod_list.extend(neighbors4)
    
    dis_info = {"发病性别倾向": gender_list, "多发群体": pop_list, "发病年龄": age_list}

    return dis_info

def filter_func(entity, dis_info, model_type):
    messages = []
    messages.append({"role":"system", "content":"你是一名专业的AI医学助手。你很擅长对疾病的知识进行整理和总结。\n"})
    gender_list = dis_info["发病性别倾向"]
    pop_list = dis_info["多发群体"]
    age_list = dis_info["发病年龄"]
    prompt = f"现在我们从各种来源获得了疾病“{entity}”的发病性别倾向、多发群体、发病年龄。但是由于知识来源过于杂乱，导致每一项知识存在很多重复或冗余的内容。\n"\
            + "以下是该疾病的相关信息：\n" \
            + f"1.发病性别倾向：{gender_list}\n"\
            + f"2.多发群体：{pop_list}\n"\
            + f"3.发病年龄：{age_list}\n"\
            + "###任务：{请你综合每一项知识的信息，对它们进行整理。你不能仅仅只是简单的罗列，当某一项知识的内容出现明显的重叠时，你需要进行总结和合并。}\n"\
            + "###提示样例：{1.若某疾病的“发病性别倾向”既包含“男性”，也包含“女性”，那么说明它的“发病性别倾向”应为“男性或女性”。\n"\
            + "2.若某疾病的“多发群体”包含“中年人”和“老年人”，那么说明它的多发群体应为“中老年群体”。\n"\
            + "3.若某疾病的“发病年龄”为“20-40岁”和“30-60岁”(即年龄范围存在重叠)，那么说明它的发病年龄应为“20-60岁”。其它情况可以以此类推。}\n"\
            + "###要求：{你只需要按照格式要求输出最后总结的结果，不要输出分析过程，也不要输出其它无关内容。如果某一项的内容为“[]”，请输出“[未知]”。}\n"\
            + "###格式：\n{1.发病性别倾向：[?]\n2.多发群体：[?]\n3.发病年龄：[?]-[?]岁}\n"
    messages.append({"role":"user", "content":prompt})
    response, history = chat_(model_type, messages)
    return response

uri = args.uri
username = args.username
# password = "Y__jFPfLmaqombcJF1Betkb5TQufxGSWEmBCOnglehM"   
password = args.password               

driver = GraphDatabase.driver(uri, auth=(username, password))
driver.verify_connectivity()
session = driver.session()

""" 知识图谱实体集合加载 """
kg_entitys = []
kg_entity_filepath = args.kg_entity_path
with open(kg_entity_filepath, 'r', encoding='utf-8') as kg_entity_file:
    kg_entity_lines = kg_entity_file.readlines()
    for kg_entity_line in kg_entity_lines:
        kg_entitys.append(kg_entity_line.strip().split('\t')[0])
kg_entity_file.close()

# ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
""" 知识图谱实体类别对应字典加载 """
fin_entity_type_map = open(args.entity_type_map_path, "r", encoding="utf-8")
entity_type_map = json.load(fin_entity_type_map)


tokenizer, chat_model = load_model(args.model_type, args.model_name_or_path)

fout = open("/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/disease_info/dis_info.json", "w", encoding="utf-8")

for i, entity in tqdm(enumerate(kg_entitys), total=len(kg_entitys), desc = "processing"):
    if entity not in entity_type_map.keys():
        continue
    entity_type = entity_type_map[entity]
    if entity_type == "疾病" or entity_type == "社会学":
        dis_info = get_disease_info(entity)
        response = filter_func(entity, dis_info, args.model_type)
        response = response.replace(":", "：")

        dis_info_dict = {}
        # 通过正则化处理response，"发病性别倾向"即为response中"发病性别倾向"和"\n"之间的内容，其它以此类推
        match1 = re.search(r'发病性别倾向：(.*?)\n', response)
        dis_info_dict["发病性别倾向"] = match1.group(1) if match1 else "[]"
        match2 = re.search(r'多发群体：(.*?)\n', response)
        dis_info_dict["多发群体"] = match2.group(1) if match2 else "[]"
        match3 = re.search(r'发病年龄：(.*?)\n', response)
        dis_info_dict["发病年龄"] = match3.group(1) if match3 else "[]"

        output_json = {entity: dis_info_dict}
        fout.write(json.dumps(output_json, ensure_ascii=False) + "\n")
        fout.flush()
    else:
        continue

fout.close()