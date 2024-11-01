from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class hyperparams:
    def __init__(self):
        self.department = ["皮肤科", "精神科", "肿瘤科"]
        self.model_version = "baichuan-7b-chat"
        self.model_name_or_path = "/home/myjia/Medical_LLM_task/LLMs_base_model/{}".format(self.model_version)
        self.input_path = "/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/KG_merge.txt"
        self.output_path = ["/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/department/pifuke_KG.txt",
                            "/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/department/jingshenke_KG.txt",
                            "/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/department/zhongliuke_KG.txt"]
        self.log_path = "/home/myjia/Medical_LLM_task/MindMap/preprocess/data_filter/department_filter.log"
        


args = hyperparams()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
chat_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map = 'auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
chat_model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
chat_model = chat_model.cuda()



def chat_baichuan(prompt):
    if len(prompt) > 2048:
        prompt = prompt[:2048]
    messages = []
    messages.append({"role": "user", "content": prompt})
    response = chat_model.chat(tokenizer, messages)
    return response

def department_filter(department_list, output_file_list, input_triples):
    prompt = "你是一个专业的医学助手，你能够准确地辨别出不同科室的知识。\n\n" \
            + "现在，我需要你帮助我判断以下知识图谱三元组属于哪个科室。\n" \
            + "三元组的格式为：{头实体\t头实体类型\t关系\t尾实体\t尾实体类型}" \
            + "该三元组可能包含疾病、药物、症状等方面的信息，请你优先从疾病角度考虑属于哪个科室。\n\n" \
            + "输入三元组：\n" + input_triples + "\n" \
            + "任务：\n{你需要根据你已掌握的医学知识，判断给你的三元组属于哪个科室(优先从疾病角度考虑)。你只需要回答科室的名称，请不要输出其他内容，也不需要给出解释。}\n\n" 
    
    result = chat_baichuan(prompt)
    for i, dep in enumerate(department_list):
        if dep in result:
            output_file_list[i].write(input_triples + "\n")
            output_file_list[i].flush()
            break
    
    return result

if __name__ == "__main__":
    log_file = open(args.log_path, "a", encoding="utf-8")
    input_file = open(args.input_path, "r", encoding="utf-8")
    output_file_list = [open(path, "a", encoding="utf-8") for path in args.output_path]
    # for out_file in output_file_list:
    #     out_file.write("head_entity\thead_type\trelation\ttail_entity\ttail_type\n")
    #     out_file.flush()
    department_list = args.department
    
    for i, line in tqdm(enumerate(input_file), total = 4385616):
        if i <= 1244094:
            continue
        result = department_filter(department_list, output_file_list, line.strip())
        log_file.write("line index: " + str(i) + "\n")
        log_file.flush()
        # print(result)
    
    input_file.close()
    for file in output_file_list:
        file.close()