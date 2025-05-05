from openai import OpenAI
import numpy as np
import json

class Config:
    llm = OpenAI(
        api_key="123",
        base_url="http://127.0.0.1:8001/v1"
    )
    embedding = OpenAI(
        api_key="xxxxx",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    chunk_size = 1000
    overlap = 200
    folder_path = "/home/cx/demo_RAG/knowledge_base"
    embedding_dim = 256
    
    embedding_top_k = 10
    bm25_top_k = 10
    rrf_top_k = 5
    reranker_path = "/home/data/ReRanker/bge-reranker-v2-m3"
    reranker_top_k = 3

# Initialize the LLM
llm = Config.llm
embedding = Config.embedding

def llm_invoke(message: str) -> str:
    """
    This function demonstrates how to invoke the LLM API.
    """
    # Example of invoking the LLM API
    result = llm.chat.completions.create(
        # model="deepseek-ai/DeepSeek-V3",
        model = "Qwen3",
        messages=[
            {
                "role": "system",
                "content": "请用轻松的语气回答问题，尽量用中文回答，除非用户要求英文。",
            },
            {
                "role": "user",
                "content": message + " /no_think",
            }
        ],
        temperature=0.1,
        top_k = 3
    )
    return result.choices[0].message.content

def read_pdf_file(file_path: str, chunk_size = 0, overlap = 0) -> str:
    """
    This function reads a PDF file and returns its content as a string.
    """
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # If chunk_size is 0, return the entire text
    if chunk_size == 0:
        return text
    # If chunk_size is specified, split the text into chunks
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def updata_database(folder_path: str, config) -> dict:

    import os

    already_files = [x.split(".")[0] for x in os.listdir(os.path.join(folder_path, "chunk_txt"))]
    files = {}
    for file in os.listdir(os.path.join(folder_path, "source_file")):
        file_name = file.split(".")[0]
        if file.endswith(".pdf") and file_name not in already_files:
            
            files[file_name] = read_pdf_file(os.path.join(folder_path, "source_file", file), config.chunk_size, config.overlap)
        elif file.endswith(".txt") and file_name not in already_files:
            pass


    for file, content in files.items():
        with open(os.path.join(folder_path, "chunk_txt", file + ".jsonl"), 'w') as f:
            for chunk in content:
                embedding_vector = embedding_invoke(chunk, config)
                json.dump({str(embedding_vector): chunk}, f,ensure_ascii=False)
                f.write("\n")

                
    return files

def embedding_invoke(text: str, config) -> list:
    """
    This function demonstrates how to invoke the embedding API.
    """
    # Example of invoking the embedding API
    result = embedding.embeddings.create(
        model="embedding-3",
        input=text,
        dimensions=config.embedding_dim
    )
    return result.data[0].embedding

if __name__ == "__main__":
    llm = OpenAI(
        api_key="123",
        base_url="http://127.0.0.1:8001/v1"
    )
    result = llm.chat.completions.create(
        model="Qwen3",
        messages=[
            {'role' : "user", "content" : "你知道吗，有论文发现，其实秦始皇并没有死，他藏在了一个兵马俑中。 /no_think"}
        ],
        temperature=0.1
    )
    print(result.choices[0].message.content)