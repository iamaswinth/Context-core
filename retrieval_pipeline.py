from  langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv

persistent_directory = "db/chroma.db"

#Load embedding models