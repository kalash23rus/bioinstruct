import requests
import json
from tqdm import tqdm
import multiprocessing
from config_loader import load_config
from database_operations import select_paragraphs, filter_processed_paragraphs, insert_qa_doc
from gpt3_operations import qa_generator
from splitter import split

# Load configuration
configs = load_config("/home/a.kalashnikov/configs/config_dbs.json")
param_dic_gpt = configs["biocadavr_gpt"]
param_dic_pmc = configs["biocadavr_pmc"]

# Set up the OpenAI API key and model ID
openai.api_key = ""
model_engine = "gpt-3.5-turbo"

# Select paragraphs from the database
query = '''
WITH extracted_paragraphs AS (
  SELECT 
    pmc_id,
    jsonb_array_elements(json_docs -> 'paragraphs') AS paragraphs
  FROM
    pmc_paragraph_json
)
SELECT 
  pmc_id,
  paragraphs
FROM
  extracted_paragraphs
WHERE
  CHAR_LENGTH(paragraphs::text) > 700;
'''
df = select_paragraphs(query, param_dic_pmc)

# Filter processed paragraphs
query = '''
SELECT DISTINCT
json_docs ->> 'context' AS paragraphs
FROM pmc_qa_set
'''
processed_paragraphs = select_paragraphs(query, param_dic_gpt)
df = filter_processed_paragraphs(df, processed_paragraphs)

# Split paragraphs
chunk_size = 30*5
chunk_overlap = 5*5
type_splitter = "nltk"
recursive_coef=5
paragraphs = []
for full_text in tqdm(df['paragraphs']):
    splited_texts = split(full_text, chunk_size, chunk_overlap, type_splitter, recursive_coef)
    paragraphs.extend(splited_texts)
paragraphs = list(set(paragraphs))

def generate_and_insert_qa(paragraph):
    try:
        qa_doc = qa_generator(paragraph, 0.0, model_engine)
        insert_qa_doc(qa_doc, "pmc_qa_set", "json_docs", param_dic_gpt)
    except:
        pass
    return "nice!"

if __name__ == '__main__':
    n_processes = 60  # Set the number of processes you want to use
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(generate_and_insert_qa, paragraphs)