from datetime import timedelta

import lancedb
import numpy as np
import pyarrow as pa
import requests
from lancedb.pydantic import LanceModel, Vector

from utils.const import *

table = None


class DocsModel(LanceModel):
    parent_source: str
    source: str
    text: str
    vector: Vector(EMBEDDINGS_DIM, pa.float16())


db = lancedb.connect("s3://warehouse/v-db/",
                     read_consistency_interval=timedelta(seconds=5))


def get_or_create_table():
    global table
    if table is None and DOCS_TABLE not in list(db.table_names()):
        return db.create_table(DOCS_TABLE, schema=DocsModel)
    if table is None:
        table = db.open_table(DOCS_TABLE)
    return table


def get_embedding(text):
    resp = requests.post(EMBEDDING_ENDPOINT,
                         json={"model": EMBEDDING_MODEL,
                               "prompt": text})
    return np.array(resp.json()["embedding"][:EMBEDDINGS_DIM], dtype=np.float16)


def search(query, limit=5):
    query_embedding = get_embedding(f"{EMBEDDING_QUERY_PREFIX}: {query}")
    res = get_or_create_table().search(query_embedding).metric("cosine").limit(limit)
    return res
