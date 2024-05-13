import nest_asyncio
nest_asyncio.apply()
import os
from dataclasses import dataclass

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load model from Azure OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser

# import
from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from IPython.display import Markdown, display
import chromadb
import streamlit as st

# Set environment variables
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-EkvrDb8xvDuEam0YOAYpsoBVBzhsLDqdd0A1QjXTg1PzSc7e"
os.environ["AZURE_OPENAI_KEY"] = "54ba524df88444caa7b5251bec720b25"
aoai_api_key = os.getenv("AZURE_OPENAI_KEY")
aoai_endpoint = "https://nef6101a.openai.azure.com/"
aoai_api_version = "2024-02-15-preview"

# Load Azure OpenAI models
embed_model = AzureOpenAIEmbedding(
    azure_deployment="text-embedding-ada-002",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
)

llm = AzureOpenAI(
    azure_deployment="gpt-35-turbo-0613",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
    temperature= 0.4
)

# Uncomment this section to init gpt4 model
# llm3 = AzureOpenAI(
#     azure_deployment="gpt-4-6013",
#     api_key=aoai_api_key,
#     azure_endpoint=aoai_endpoint,
#     api_version=aoai_api_version,
#     temperature= 0.4
# )

# change to llm3 to set used model to gpt4
Settings.llm = llm

# llm2 = OpenAI(model="gpt-3.5-turbo-0125")

# Load from disk
db_instruct = chromadb.PersistentClient(path="./chroma_db_Instruct")
chroma_collection_instruct = db_instruct.get_or_create_collection("quickstart2")
vector_store_instruct = ChromaVectorStore(chroma_collection=chroma_collection_instruct)
# storage_context_instruct = StorageContext.from_defaults(vector_store=vector_store_instruct)

# Query Data from the persisted index
index_instruct = VectorStoreIndex.from_vector_store(vector_store=vector_store_instruct, embed_model=embed_model, similarity_top_k=10)
query_engine_instruct = index_instruct.as_query_engine()
# response = query_engine_instruct.query("22.1.1 Types of Referring Expressions")
# display(Markdown(f"<b>{response}</b>"))

st.title("JurafskyGPT")

@dataclass
class Message:
    actor: str
    payload: str

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

# if "messages" not in st.session_state:
#     st.session_state.messages = []
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi!How can I help you?")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)
    
# Accept user input
prompt = st.text_input("Ask")
if prompt:
    # # Add user message to chat history
    # st.session_state.messages.append(prompt)
    # # Display user message in chat message container
    # st.write("User:\n", prompt)
    # # Display assistant response in chat message container
    # assist_response = query_engine_instruct.query(prompt)
    # st.write("Assistant:\n", assist_response.response)
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    response: str = query_engine_instruct.query(prompt).response
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)