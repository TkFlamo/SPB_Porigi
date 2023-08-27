import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import time
import pickle
import faiss
import json

def vector_bert(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    return model_output.last_hidden_state[0][0]

def model(query, full_vectors_dict, model, tokenizer):
    query_vector = vector_bert(query, model, tokenizer).numpy()
    vectors_matrix = np.vstack([vector for vector in full_vectors_dict.values()])
    index = faiss.IndexFlatL2(vectors_matrix.shape[1])
    index.add(vectors_matrix)
    distance, indices = index.search(query_vector.reshape(1, -1), 10)
    similar_indices = indices[0]
    top = list(full_vectors_dict.keys())[similar_indices[0]]
    top10 = [list(full_vectors_dict.keys())[i] for i in similar_indices]
    return top10
    
    