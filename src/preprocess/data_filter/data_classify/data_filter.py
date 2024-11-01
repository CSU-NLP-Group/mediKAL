from transformers import AutoTokenizer, AutoModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()
from tqdm import tqdm
import json


def chat(prompt):
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   do_sample=False,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

def final_answer(query, response):
    prompt = "你是一名优秀的AI医生,你可以根据提供的医疗对话样例判断该样例属于哪个科室。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"query：{query},\n\n response：{response}\n\n" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

if __name__ == '__main__':
    fin = open("/home/myjia/Medical_LLM_task/dataset/ChatMed_Consult_Dataset/ChatMed_Consult-v0.3.json", 'r', encoding='utf-8')
    fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/ChatMed_Consult_Dataset/ChatMed_Consult-v0.3_pifuke.json", 'w', encoding='utf-8')
    fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/ChatMed_Consult_Dataset/ChatMed_Consult-v0.3_jingshenke.json", 'w', encoding='utf-8')
    fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/ChatMed_Consult_Dataset/ChatMed_Consult-v0.3_zhongliuke.json", 'w', encoding='utf-8')
    # 按行读取json文件
    for i, line in tqdm(enumerate(fin)):
        line = json.loads(line)
        query = line['query']
        response = line['response']
        result = final_answer(query, response)
        print("预测结果：{}\n".format(result))
        if "皮肤科" in result:
            fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        elif "精神科" in result:
            fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        elif "肿瘤科" in result:
            fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
