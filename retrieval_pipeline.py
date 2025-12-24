from  langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

#Load embedding models
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query = "Who founded Alphabet, what are the products that are make by alphabet"

# retriever = db.as_retriever(search_kwargs = {"k":3})

retriever = db.as_retriever(
    search_type = "similarity_score_threshold", 
    search_kwargs={
        "k":5,
        "score_threshold":0.3 #only return chunks that has cosine simularity >= 0.3
    }
)

relevent_docs = retriever.invoke(query)

print(f"User query: {query}")

print("--Contents--")

for i, doc in enumerate(relevent_docs, 1):
    print(f"Document {i}: \n{doc.page_content}")

#conbine the query and the revelent documents

conbined_imputs = f"""
Based on the following documents, please answer this question:{query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevent_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

#create open ai model

model = ChatOpenAI(
    model="gpt-4o"
)

#Define the message for the assistent
messages = [
    SystemMessage(content="You are a helpfull assistent."),
    HumanMessage(content=conbined_imputs)
]

result = model.invoke(messages)

print("\n --- Generated contents ---")

print("Contents only:")

print(result.content)