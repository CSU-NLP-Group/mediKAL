import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import openai
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import re

tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm2-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm2-6b", trust_remote_code=True).cuda()
chat_model.eval()

def chat(prompt):
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

f = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/zhongliuke_KG.txt", "r")
fout = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_based_docs/zhongliuke_KG_docs.json", "w")

def prompt_doc_generate(triplet_cluster):
    prompt = f"""
    There are some knowledge graph triplets. They follow entity-relationship-entity format.
    \n\n
    {triplet_cluster}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as sentence1, sentence2,...\n\n

    Output:
    """

    response = chat(prompt=prompt)
    return response


cur_target_entity = ""
cur_triplet_cluster = ""
for line in tqdm(f.readlines()):
    line = line.strip().split("\t")
    triplet = line[0] + "-" + line[1] + "-" + line[2]
    if line[0] == cur_target_entity or line[2] == cur_target_entity:
        cur_triplet_cluster += triplet + "\n"
    else:
        if cur_triplet_cluster != "":
            response = prompt_doc_generate(cur_triplet_cluster)
            document = ""
            if "\n\n" in response:
                if "：" in response:
                    response = re.split('：|\n\n', response)
                else:
                    response = re.split(':|\n\n', response)
            else:
                if "：" in response:
                    response = re.split('：|\n', response)
                else:
                    response = re.split(":|\n", response)
            document = ""
            for i in range(len(response)):
                if i % 2 == 1:
                    document += response[i]
            fout.write(json.dumps({"key_entity": cur_target_entity, "document": document}, ensure_ascii=False))
            fout.write("\n")
            fout.flush()
            # document_value = response.strip().split("输出: ")[1]
            # fout.write(json.dumps({"key_entity": cur_target_entity, "document": document_value}, ensure_ascii=False))
            # fout.flush()
            
        cur_target_entity = line[0]
        cur_triplet_cluster = triplet + "\n"
    

f.close()