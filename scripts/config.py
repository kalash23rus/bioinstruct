import json

with open("./config_dbs.json") as file:
    configs = json.load(file)

param_dic_gpt = configs["biocadavr_gpt"]
param_dic_pmc = configs["biocadavr_pmc"]

openai_api_key = "s"
model_engine = "gpt-3.5-turbo"