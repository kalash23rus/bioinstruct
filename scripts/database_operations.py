import pandas as pd
from db_tools import insert_dictionary_into_postgres, select_to_df

def select_paragraphs(query, param_dic_pmc):
    df = select_to_df(query, param_dic_pmc)
    return df

def filter_processed_paragraphs(df, processed_paragraphs):
    df = df[~df["paragraphs"].isin(processed_paragraphs["paragraphs"])]
    return df

def insert_qa_doc(qa_doc, table_name, column_name, param_dic_gpt):
    insert_dictionary_into_postgres(qa_doc, table_name, column_name, param_dic_gpt)