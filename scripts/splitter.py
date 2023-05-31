import pandas as pd
import numpy as np
from langchain.text_splitter import TokenTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter

def split(full_text, chunk_size, chunk_overlap, type_splitter, recursive_coef=5):
    if type_splitter == 'token':
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 

    elif type_splitter == 'nltk':
        text_splitter = NLTKTextSplitter(chunk_size=chunk_size*recursive_coef, chunk_overlap=chunk_overlap*recursive_coef)

    elif type_splitter == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size*recursive_coef,
            chunk_overlap=chunk_overlap*recursive_coef,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    else:
        raise TypeError("bad type of splitter.\tPlease use from list [token, nltk, recursive]")

    texts = text_splitter.split_text(full_text)
    return texts