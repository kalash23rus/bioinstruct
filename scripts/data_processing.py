import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def create_models(model_names):
    models = {}
    for name in model_names:
        model_key = name.replace('-', '_')
        models[model_key] = SentenceTransformer(name)
    return models

def create_embeddings(df, column_name, model, batch_size=10000):
    df_copy = df.copy()
    embeddings_column_name = f"{column_name}_embeddings"
    
    num_rows = df_copy.shape[0]
    num_batches = int(np.ceil(num_rows / batch_size))
    
    embeddings = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_rows)
        
        batch_texts = df_copy[column_name].iloc[start_idx:end_idx].tolist()
        batch_text_embeddings = model.encode(batch_texts)
        
        embeddings.extend(batch_text_embeddings)
    
    df_copy[embeddings_column_name] = pd.Series(embeddings, index=df_copy.index)
    
    return df_copy