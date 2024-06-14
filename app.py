import json
import multiprocessing
import random
import time
import urllib.parse

import gradio as gr
import pandas as pd
import requests
import s3fs
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel

from utils.const import *
from utils.doc_process_util import split_doc_by_chunks
from utils.vector_util import get_embedding, get_or_create_table, search

# using fsspec configuration to connect to minio
# https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration
# sample configuration below
# {
#     "s3":
#         {
#             "key": "minioadmin",
#             "secret": "minioadmin",
#             "client_kwargs": {
#                 "endpoint_url": "http://localhost:9000"
#             }
#         }
# }
s3 = s3fs.S3FileSystem()

context_df = []

# multiprocessing queues to handle data
add_data_queue = multiprocessing.Queue()
delete_data_queue = multiprocessing.Queue()


def add_vector_job():
    data = []
    table = get_or_create_table()

    while not add_data_queue.empty():
        item = add_data_queue.get()
        data.append(item)

    if len(data) > 0:
        df = pd.DataFrame(data)
        table.add(df)
        table.compact_files()
        print(f"Total Rows Added: {len(table.to_pandas())}")


def delete_vector_job():
    table = get_or_create_table()
    source_data = []
    while not delete_data_queue.empty():
        item = delete_data_queue.get()
        source_data.append(item)
    if len(source_data) > 0:
        filter_data = ", ".join([f'"{d}"' for d in source_data])
        table.delete(f'source IN ({filter_data})')
        table.compact_files()
        table.cleanup_old_versions()
        print(f"Total Rows Deleted: {len(table.to_pandas())}")


scheduler = BackgroundScheduler()

scheduler.add_job(add_vector_job, 'interval', seconds=10)
scheduler.add_job(delete_vector_job, 'interval', seconds=10)

app = FastAPI()


class WebhookData(BaseModel):
    data: dict


@app.on_event("startup")
async def startup_event():
    get_or_create_table()
    if not scheduler.running:
        scheduler.start()


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()


@app.post("/api/v1/document/notification")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    json_data = await request.json()
    print(json.dumps(json_data, indent=2))
    if json_data["EventName"] == "s3:ObjectCreated:Put":
        print("New object created!")
        background_tasks.add_task(create_object_task, json_data)
    if json_data["EventName"] == "s3:ObjectRemoved:Delete":
        print("Object deleted!")
        background_tasks.add_task(delete_object_task, json_data)
    # print(json.dumps(json_data, indent=2))
    return {"status": "success"}


@app.post("/api/v1/metadata/notification")
async def receive_metadata_webhook(request: Request, background_tasks: BackgroundTasks):
    json_data = await request.json()
    print(json.dumps(json_data, indent=2))
    if json_data["EventName"] == "s3:ObjectCreated:Put":
        print("New Metadata created!")
        background_tasks.add_task(create_metadata_task, json_data)
    if json_data["EventName"] == "s3:ObjectRemoved:Delete":
        print("Metadata deleted!")
        background_tasks.add_task(delete_metadata_task, json_data)
    return {"status": "success"}


def create_metadata_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])
        print(bucket_name,
              object_key)
        with s3.open(f"{bucket_name}/{object_key}", "r") as f:
            data = f.read()
            chunk_json = json.loads(data)
            embeddings = get_embedding(f"{EMBEDDING_DOCUMENT_PREFIX}: {chunk_json['page_content']}")
            add_data_queue.put({
                "text": chunk_json["page_content"],
                "parent_source": chunk_json.get("metadata", "").get("source", ""),
                "source": f"{bucket_name}/{object_key}",
                "vector": embeddings
            })
    return "Task Completed!"


def delete_metadata_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])
        delete_data_queue.put(f"{bucket_name}/{object_key}")
    return "Task completed!"


def create_object_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])
        print(record["s3"]["bucket"]["name"],
              record["s3"]["object"]["key"])

        doc_splits = split_doc_by_chunks(bucket_name, object_key)

        for i, chunk in enumerate(doc_splits):
            print(chunk.json())
            source = f"warehouse/{METADATA_PREFIX}/{bucket_name}/{object_key}/chunk_{i:05d}.json"
            with s3.open(source, "w") as f:
                f.write(chunk.json())
    return "Task completed!"


def delete_object_task(json_data):
    for record in json_data["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = urllib.parse.unquote(record["s3"]["object"]["key"])
        s3.delete(f"warehouse/{METADATA_PREFIX}/{bucket_name}/{object_key}", recursive=True)
    return "Task completed!"


def llm_chat(user_question, history):
    history = history or []
    global context_df
    user_message = f"**You**: {user_question}"
    res = search(user_message)
    documents = " ".join([d["text"].strip() for d in res.to_list()])
    context_df = res.to_pandas()
    context_df = context_df.drop(columns=['source', 'vector'])
    bot_response = "**AI:** "
    llm_resp = requests.post(LLM_ENDPOINT,
                             json={"model": LLM_MODEL,
                                   "messages": [
                                       {"role": "user",
                                        "content": RAG_PROMPT.format(user_question=user_question, documents=documents)
                                        }
                                   ],
                                   "options": {
                                       "temperature": 0,
                                       "top_p": 0.90,
                                   }},
                             stream=True)
    for resp in llm_resp.iter_lines():
        json_data = json.loads(resp)
        bot_response += json_data["message"]["content"]
        yield bot_response


def sidebar_event():
    events = ["Event 1", "Event 2", "Event 3", "Event 4", "Event 5"]
    yield gr.update(f"<div>Simulated Event: {random.choice(events)}</div>")
    time.sleep(3)


def progress_bar(num_steps):
    progress = gr.Progress()
    progress(0, desc="Starting...")
    for i in progress.tqdm(range(num_steps)):
        print(i)
        time.sleep(0.5)
    return "Progress bar completed!"


def clear_events():
    global context_df
    context_df = []
    return context_df


with gr.Blocks(gr.themes.Soft()) as demo:
    gr.Markdown("## RAG with MinIO")
    ch_interface = gr.ChatInterface(llm_chat, undo_btn=None, clear_btn="Clear")
    ch_interface.chatbot.height = 600
    ch_interface.chatbot.show_label = False
    gr.Markdown("### Context Supplied")
    context_dataframe = gr.DataFrame(headers=["parent_source", "text", "_distance"], wrap=True)
    # ch_interface.chatbot.likeable = True
    ch_interface.clear_btn.click(clear_events, [], context_dataframe)


    @gr.on(ch_interface.output_components, inputs=[ch_interface.chatbot], outputs=[context_dataframe])
    def update_chat_context_df(text):
        global context_df
        if context_df is not None:
            return context_df
        return ""

if __name__ == "__main__":
    demo.queue()
    app = gr.mount_gradio_app(app, demo, path=CHAT_API_PATH)
    uvicorn.run(app, host="0.0.0.0", port=8808)
