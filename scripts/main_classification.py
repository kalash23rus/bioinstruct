import pandas as pd
import openai
from config import openai_api_key, model_engine, param_dic_gpt, param_dic_pmc
from data_processing import create_models, create_embeddings
from model_training import train_and_evaluate_models
from prediction import predict_with_models

# Set up the OpenAI API key and model ID
openai.api_key = openai_api_key

# Load data
df_train = pd.read_csv("data/train.csv")
df_train = df_train.drop_duplicates()
df_train["num_labels"] = df_train["text_labels"].map({"bad":1,"good":0})

# Create models and embeddings
sentence_transformer_models = ['paraphrase-mpnet-base-v2']
models = create_models(sentence_transformer_models)
df_train = create_embeddings(df_train, 'text', models['paraphrase_mpnet_base_v2'])

# Train and evaluate models
X = df_train['text_embeddings'].tolist()
y = df_train['num_labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
leaderboard, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)