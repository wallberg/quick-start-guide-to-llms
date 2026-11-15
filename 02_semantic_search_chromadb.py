# %% [markdown]
# Semantic search using chromadb for the vector database, using
# built-in embeddings

# %%
import hashlib
import logging
import os
from datetime import datetime, timezone

import chromadb
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# %%
CHROMADB_COLLECTION = 'semantic-search-test'

chromadb_client = chromadb.PersistentClient(path="data/my_chroma_db")

# Store the index as a variable
# index = pc.Index(name=INDEX_NAME)
collection = chromadb_client.get_or_create_collection(name=CHROMADB_COLLECTION)


# %%
def my_hash(s):
    # Return the MD5 hash of the input string as a hexadecimal string
    return hashlib.md5(s.encode()).hexdigest()

my_hash('I love to hash it')

# %%
def prepare_for_pinecone(texts):
    # Get the current UTC date and time
    now = datetime.now(timezone.utc).isoformat()

    # Create tuples of (hash, embedding, metadata) for each input string and its corresponding vector embedding
    # The my_hash() function is used to generate a unique hash for each string, and the datetime.utcnow() function is used to generate the current UTC date and time
    return [
        (
            my_hash(text),  # A unique ID for each string, generated using the my_hash() function
            text,
            {'date_uploaded': now}  # A dictionary of metadata, including the original text and the current UTC date and time
        )
        for text in texts
    ]


# %%
# def upload_texts_to_pinecone(texts, namespace=NAMESPACE, batch_size=None, show_progress_bar=False):
def upload_texts_to_pinecone(texts, batch_size=None, show_progress_bar=False):
    # Call the prepare_for_pinecone function to prepare the input texts for indexing
    total_upserted = 0
    if not batch_size:
        batch_size = len(texts)

    _range = range(0, len(texts), batch_size)
    for i in tqdm(_range) if show_progress_bar else _range:
        batch = texts[i: i + batch_size]
        prepared_texts = prepare_for_pinecone(batch)

        collection.upsert(
            documents=[document for _, document, _ in prepared_texts],
            metadatas=[metadata for _, _, metadata in prepared_texts],
            ids=[_id for _id, _, _ in prepared_texts],
        )
        total_upserted += len(prepared_texts)

    return total_upserted


# %%
def query_from_pinecone(query, top_k=3, include_metadata=True):

    return collection.query(
      query_texts=[query],
      n_results=top_k,
    )

# %%
# def delete_texts_from_pinecone(texts, namespace=NAMESPACE):
def delete_texts_from_pinecone(texts):

    # Compute the hash (id) for each text
    hashes = [hashlib.md5(text.encode()).hexdigest() for text in texts]

    # The ids parameter is used to specify the list of IDs (hashes) to delete
    return collection.delete(ids=hashes)

# %%
from datasets import load_dataset

os.environ["HF_DATASETS_OFFLINE"] = "1"
dataset = load_dataset("google/xtreme", "MLQA.en.en")

# rename test -> train and val -> test (as we will use it in later in this chapter)
dataset['train'] = dataset['test']
dataset['test'] = dataset['validation']
del dataset['validation']


# %%
unique_passages = list(set(dataset['test']['context']))
for idx in tqdm(range(0, len(unique_passages), 32)):
    passages = unique_passages[idx:idx + 32]
    upload_texts_to_pinecone(passages)


# %%
print(collection.count())

# %%
result = query_from_pinecone('Does an infection for Sandflies go away over time?')
for _id, document, distance in zip(result['ids'][0], result['documents'][0], result['distances'][0]):
    print(_id)
    print(document)
    print(distance)
    print()
