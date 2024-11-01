from transformers import AutoTokenizer, AutoModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()
from tqdm import tqdm
import json


def chat(prompt):
    if len(prompt) > 1500:
        prompt = prompt[:1500]
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   do_sample=False,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

def final_answer(query, response):
    prompt = "你是一名优秀的AI医生,你可以根据所提供的医疗知识库的问答对样例，判断该样例所提供的信息对于哪个科室的疾病具有借鉴意义。请注意，所给的问答对的信息可能不是与某类疾病直接相关，请你仔细辨别\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"question：{query},\n\n answer：{response}\n\n" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

if __name__ == '__main__':
    fin = open("/home/myjia/Medical_LLM_task/dataset/huatuo-26m/医疗数据/Knowledge_bases.txt", 'r', encoding='utf-8')
    fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/huatuo-26m/医疗数据/Knowledge_bases_pifuke.txt", 'w', encoding='utf-8')
    fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/huatuo-26m/医疗数据/Knowledge_bases_jingshenke.txt", 'w', encoding='utf-8')
    fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/huatuo-26m/医疗数据/Knowledge_bases_zhongliuke.txt", 'w', encoding='utf-8')
    # 按行读取json文件
    for i, line in tqdm(enumerate(fin)):
        # line = json.loads(line)
        # query = line['query']
        # response = line['response']
        newline = line.strip().split('\t')
        question = newline[0]
        answer = newline[1]
        result = final_answer(question, answer)
        print("预测结果：{}\n".format(result))
        if "皮肤科" in result: # 写入新的txt文件
            fout_pifuke.write(line)
            fout_pifuke.flush()
        elif "精神科" in result or "心理内科" in result:
            fout_jingshenke.write(line)
            fout_jingshenke.flush()
        elif "肿瘤科" in result:
            fout_zhongliuke.write(line)
            fout_zhongliuke.flush()