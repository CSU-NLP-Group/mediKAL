from tqdm import tqdm
import json
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-PV3sbyymCK32122KuYfDyWqakFO3teI37XV1hqM8XC0IHc4S",
    base_url="https://api.chatanywhere.tech/v1"
)

def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

fin = open('/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/dialogue/new_data_KG/diseases.json', 'r', encoding='utf-8')

# messages = [{'role': 'user','content': '鲁迅和周树人的关系'},]

for i, line in tqdm(enumerate(fin)):
    dis_dict = json.loads(line)
    core_dis = dis_dict['核心词']
    prompt = "我们将会为你提供一组以某个疾病为核心词的一系列医学知识库中的知识。\n" \
            + "请你围绕上述知识，构造一组医学诊疗对话。\n" \
            + "以下是一个示例：\n" \
            + "#输入疾病知识: \n"\
            + '{"中心词": "脂溢性皮炎", "病因": ["皮脂溢出", "微生物", "B族维生素缺乏", "精神因素", "饮食习惯"], "临床表现": ["红斑", "鳞屑", "瘙痒", "丘疹"], "治疗": ["B族维生素", "糖皮质激素", "规律作息", "硫化硒洗剂"]}\n\n'\
            + "#输出构建的问答对: \n"\
            + '{"病人: 医生，我最近感觉头皮很油，用手挠会有很多头皮屑。", "医生：可能是脂溢性皮炎，你最近有熬夜吗，饮食怎么样？", "病人：最近作息确实不太规律，而且饮食也比较油腻。", "医生：建议规律作息，补充一些B族维生素，必要的话可以外用硫化硒洗剂。"}\n\n' \
            + f"现在, 请根据以下知识, 按照上面提供的要求, 仿照示例的格式进行输出。请注意，你不需要在一轮对话中包含所有的知识，如果提供给你的知识太多，你可以创建多个不同的对话，或者构造多轮对话。\n"\
            + "#输入：\n"\
            + f"{line}" \
            + "#输出：\n"\
            + "{}"
    messages = [{'role': 'system','content': '你是一个专业的皮肤科医生，同时是一位经验丰富的数据工程师。\n'},]
    messages.append({'role': 'user','content': prompt})
    print(messages)
    qa_pairs = gpt_35_api(messages)
            
