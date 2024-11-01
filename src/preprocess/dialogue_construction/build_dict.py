import json
from neo4j import GraphDatabase, basic_auth
from tqdm import tqdm

fin = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/entity_type_maps/entity_type_map_pifuke_small.json", "r", encoding="utf-8")
# fin是一个大的json字典
input_data = json.load(fin)

class hyperparams:
    def __init__(self):
        
        self.uri = "bolt://localhost:7687"
        self.username = "neo4j"
        self.password = "123456"
        


args = hyperparams()

uri = args.uri
username = args.username
# password = "Y__jFPfLmaqombcJF1Betkb5TQufxGSWEmBCOnglehM"   

password = args.password                  
driver = GraphDatabase.driver(uri, auth=(username, password))
driver.verify_connectivity()
session = driver.session()

# 创建一个空字典来保存结果
disease_dict = {}

# 遍历输入数据中的每个键
# 创建一个空字典来保存结果
disease_dict = {}

# 遍历输入数据中的每个键(给这个遍历过程加进度条)
for key in tqdm(input_data):
    if input_data[key] == "疾病":
        # 执行查询来获取与当前疾病节点相关的所有边
        result = session.run("MATCH (d:`疾病` {name: $name})-[r]->(n) RETURN type(r) as relation, n.name as name", name=key)
        
        # 创建一个空字典来保存当前疾病的相关信息
        disease_info = {"核心词": key}
        
        # 遍历查询结果
        for record in result:
            # 获取关系类型和节点名称
            relation = record['relation']
            name = record['name']
            
            # 如果当前关系类型还没有在字典中，就添加一个空列表
            if relation not in disease_info:
                disease_info[relation] = []
            
            # 将节点名称添加到对应的列表中
            disease_info[relation].append(name)
        
        # 将当前疾病的相关信息添加到结果字典中
        disease_dict[key] = disease_info

# 打开文件并写入查询结果
with open('/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/dialogue/new_data_KG/diseases.json', 'w') as f:
    for key in disease_dict:
        f.write(json.dumps(disease_dict[key], ensure_ascii=False) + '\n')