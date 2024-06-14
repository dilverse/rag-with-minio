from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import S3FileLoader

from utils.const import *

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=64,
                                               length_function=len)


def get_text_splitter():
    return text_splitter


def split_doc_by_chunks(bucket_name, object_key):
    loader = S3FileLoader(bucket_name,
                          object_key,
                          endpoint_url=MINIO_ENDPOINT,
                          aws_access_key_id=MINIO_ACCESS_KEY,
                          aws_secret_access_key=MINIO_SECRET_KEY)
    docs = loader.load()
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits
