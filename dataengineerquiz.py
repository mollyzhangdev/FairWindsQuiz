import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import sys
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

CHUNK_SIZE = 100
MAXIMUM_UPLOAD_SIZE = 2 * 1024 * 1024
url = "https://data.cityofnewyork.us/resource/uvpi-gqnh.csv"
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=pinecone_api_key)
df_origin = pd.read_csv(url, header=0)
index_name = "fairwinds"

### Question 1. Extract a CSV File from a Public Repository and Load into a DataFrame
logging.info("Printing the first 5 rows of the dataset:")
print(df_origin.head(5))


### Question 3. Convert a DataFrame to a Free Vector Database Using Chunking
def generate_embeddings(df):

    global numeric_cols, text_cols

    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[text_cols] = df[text_cols].astype(str).fillna("")
    df["combined_text"] = df[text_cols].agg(" ".join, axis=1)
    df["text_embedding"] = df["combined_text"].apply(lambda x: model.encode(x, show_progress_bar=False).tolist())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def create_index():
    embedding_dim = model.get_sentence_embedding_dimension()
    numeric_feature_dim = len(numeric_cols)
    actual_dimension = embedding_dim + numeric_feature_dim

    if index_name not in pc.list_indexes().names():
        logging.info(f"Creating Pinecone index with {actual_dimension} dimensions...")
        pc.create_index(name=index_name, dimension=actual_dimension, metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

    index = pc.Index(index_name)
    return index

def create_chunks(df):
    chunks = []
    for start in range(0, len(df), CHUNK_SIZE):
        vectors = []
        end = min(start + CHUNK_SIZE, len(df))
        chunk = df.iloc[start:end]
        
        for _, row in chunk.iterrows():
            unique_id = row["tree_id"]
            row_vector = row["text_embedding"]

            numeric_values = [float(row[col]) for col in numeric_cols]
            row_vector.extend(numeric_values)
            metadata = {col: row[col] for col in text_cols}
            metadata.update({col: row[col] for col in numeric_cols})
            vectors.append((str(unique_id), row_vector, metadata))

        chunks.append(vectors)
    return chunks

def upload_to_db(chunks, index):
    total_uploaded = 0
    for chunk in chunks:
        if sys.getsizeof(chunk) < MAXIMUM_UPLOAD_SIZE:
            size = len(chunk)
            logging.info(f"Uploading {size} vectors to Pinecone")

            try:
                index.upsert(chunk)
                logging.info(f"Successfully uploaded {size} vectors")
                total_uploaded += size

            except Exception as e:
                logging.error(f"Failed to upload batch: {e}")
        else:
            logging.error("Chunk too large, reducing chunk size.")

    logging.info(f"All chunks processed successfully! Total records uploaded: {total_uploaded}")

## Generate embeddings for all text and numeric columns
df = generate_embeddings(df_origin)

## Create chunks for all 1000 records with chunk size 100
chunks = create_chunks(df)

## Create index
index = create_index()

## Update to Pinecone database
upload_to_db(chunks, index)

time.sleep(5)

index = pc.Index(index_name, metric="cosine")

## Retrieve and display sample embeddings from the database
logging.info("Retrive the 15th row from the database by ID")
query_results = index.fetch(ids=["189834"])
print(query_results.vectors)

