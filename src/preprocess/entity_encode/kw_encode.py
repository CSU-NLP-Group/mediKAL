from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from tqdm import tqdm
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model = SentenceTransformer('/home/myjia/Medical_LLM_task/PLMs_base_model/all-mpnet-base-v2')
model.to("cuda")

keshi = "jingshenke"
df_input = pd.read_excel("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/{}_with_keywords.xlsx".format(keshi))

# df_input = df_input.rename(columns={'关键词': 'keywords_cmekg', 'keywords': 'keywords_LLM'})
all_keywords = set([])
for index, row in tqdm(df_input.iterrows(), total = df_input.shape[0]):
    keywords = row['keywords_cmekg']
    # keywords_old = row['keywords_LLM']
    keywords = keywords.split(",")
    all_keywords.update(keywords)
all_keywords = list(all_keywords)
embeddings = model.encode(all_keywords, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": all_keywords,
    "embeddings": embeddings,
}

with open(f"/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/embeddings/{keshi}_keyword_cmekg_embeddings.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)