import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from tqdm import tqdm
import openai
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from db_tools import (
    insert_dictionary_into_postgres,
    select_to_df,
    insert_values_to_table,
    execute_values,
)

# Set up the OpenAI API key and model ID
openai.api_key = ""
model_engine = "gpt-3.5-turbo"

# Load configuration file
with open("./config_dbs.json") as file:
    configs = json.load(file)

param_dic_gpt = configs["biocadavr_gpt"]
param_dic_pmc = configs["biocadavr_pmc"]