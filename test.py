import pandas as pd
import pickle

df = pd.read_csv("analysis/1229/model_doc2vec/patent_all_sim_matrix.csv", encoding='utf-8-sig')
with open('analysis/1229/model_doc2vec/similarity.pickle', mode='wb') as f:
    pickle.dump(df, f)

