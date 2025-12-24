import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path = "docs"):
    print(f"Loading documents from the {docs_path}...")

    #check if the file directrory exist
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create one")
    
    #load all the text file from the directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    document = loader.load()    

    if len(document) == 0:
        raise FileNotFoundError(f"No text in the directory {docs_path} does not exist. Please create one")
    
    for i, doc in enumerate(document[:2]):
        print(f"\nDocument {i+1}")
        print(f"  Source :{doc.metadata['source']}")
        print(f" Content Length: {len(doc.page_content)} Characters")
        print(f" Content Preview: {doc.page_content[:100]}....")
        print(f" metadata: {doc.metadata}")

    return document


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n Chunk {i+1}")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Lenght: {len(chunk.page_content)}")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("--"*50)

    #     if(len(chunks) > 5):
    #         print("There are more than 5 Chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    print("Create Embedding and storing in vector DB")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Create vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Finished creating vector store")

    print(f"Vector store created. Saved to {persist_directory}")

    return vectorstore



def main():
    print("Main Function")

    #load all the file
    documents = load_documents(docs_path="docs")

    #chunk all the file
    chunks = split_documents(documents)

    #store it in the vector database
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()