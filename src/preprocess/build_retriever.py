# import json
# from tqdm import tqdm
# fin = open("/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_merge.txt", "r", encoding="utf-8")
# lines = fin.readlines()

# fout = open("/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_merge.jsonl", "w", encoding="utf-8")
# for i, line in tqdm(enumerate(lines)):
#     line_dict = {}
#     line = line.strip().split("\t")
#     word = line[0]
#     id = line[1]
#     line_dict['id'] = id
#     line_dict['text'] = word
#     json.dump(line_dict, fout, ensure_ascii=False)
#     fout.write("\n")
# fout.close()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from retriv import HybridRetriever, DenseRetriever
# HybridRetriever.delete("hr_bm25_bge")
# DenseRetriever.delete("dr_corom_emb")
# dr: "dr_corom_emb",  "dr_bge_emb"
# hr: "hr_bm25_corom", "hr_bm25_bge", 

dr = DenseRetriever.load("/home/myjia/.retriv/collections/dr_bge_emb")

dr.search(
  query="高血压",    # What to search for        
  return_docs=True,          # Default value, return the text of the documents
  cutoff=5,                # Default value, number of results to return
)

print(dr.search(
  query="高血压",    # What to search for        
  return_docs=True,          # Default value, return the text of the documents
  cutoff=5,                # Default value, number of results to return
))
# dr = DenseRetriever(
#   index_name="dr_bge_emb",
#   model="/home/myjia/Medical_LLM_task/PLMs_base_model/bge-large-zh-v1.5",
#   normalize=True,
#   max_length=128,
#   use_ann=True,
# )


# dr = dr.index_file(
#   path="/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_merge.jsonl",  # File kind is automatically inferred
#   embeddings_path=None,       # Default value
#   use_gpu=True,              # Default value
#   batch_size=512,             # Default value
#   show_progress=True,         # Default value
#   callback=lambda doc: {      # Callback defaults to None.
#     "id": doc["id"],
#     "text": doc["text"], }         
#   )

# from retriv import HybridRetriever

# hr = HybridRetriever(
#     # Shared params ------------------------------------------------------------
#     index_name="hr_bm25_bge",
#     # Sparse retriever params --------------------------------------------------
#     sr_model="bm25",
#     min_df=1,
#     tokenizer=None,
#     stemmer=None,
#     stopwords=None,
#     do_lowercasing=False,
#     do_ampersand_normalization=True,
#     do_special_chars_normalization=True,
#     do_acronyms_normalization=True,
#     do_punctuation_removal=True,
#     # Dense retriever params ---------------------------------------------------
#     dr_model="/home/myjia/Medical_LLM_task/PLMs_base_model/bge-large-zh-v1.5",
#     normalize=True,
#     max_length=128,
#     use_ann=True,
# )

# hr = hr.index_file(
#   path="/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_merge.jsonl",  # File kind is automatically inferred
#   embeddings_path=None,       # Default value
#   use_gpu=True,              # Default value
#   batch_size=512,             # Default value
#   show_progress=True,         # Default value
#   callback=lambda doc: {      # Callback defaults to None.
#     "id": doc["id"],
#     "text": doc["text"],
#   }
# )