import os
import json
import numpy as np
import faiss
import jieba
import ast
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker


import prompt
import utils

def init_database(config : utils.Config) -> tuple:
    """
    Initialize the database by loading the chunked text files and their embeddings.
    Args:
        config (utils.Config): Configuration object containing folder path and embedding dimensions.
    Returns:
        tuple: A tuple containing the database index (embeddings) and the database content (text chunks).
    """
    utils.updata_database(config.folder_path, config)
    database_index = []
    database_content = []
    for file in os.listdir(os.path.join(config.folder_path, "chunk_txt")):

        file_name = file.split(".")[0] + "_"
        
        with open(os.path.join(config.folder_path, "chunk_txt", file), 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                embedding_vector = ast.literal_eval(list(data.keys())[0])
                chunk = list(data.values())[0]

                database_index.append(embedding_vector)
                database_content.append({file_name + str(i): chunk})
    
    database_index = np.array(database_index)
    # database_content:
    # [{'file_0': 'text chunk 1'}, {'file_1': 'text chunk 2'}, ...]
    return database_index, database_content
            
def BM25_search(user_query, database_content: list[dict], config) -> tuple:
    user_query = jieba.lcut(user_query)
    tokenized_cropus = [jieba.lcut(list(docs.values())[0]) for docs in database_content]
    bm25 = BM25Okapi(tokenized_cropus)
    score = bm25.get_scores(user_query)
    top_k_indices = np.argsort(score)[-config.bm25_top_k:][::-1]
    top_k_scores = score[top_k_indices]
    
    return top_k_indices, top_k_scores
    


def embedding_search(user_query, database_index, config) -> tuple:
    """
    Perform embedding search using FAISS.
    Args:
        user_query (str): The query string to search for.
        database_index (list): The list of database embeddings.
        config (utils.Config): Configuration object containing embedding dimensions.
        k (int): The number of nearest neighbors to retrieve.
    Returns:
        tuple: Distances and indices of the nearest neighbors.
    """
    faiss_index = faiss.IndexFlatL2(config.embedding_dim)
    faiss_index.add(database_index)
    # faiss.write_index(faiss_index, os.path.join(config.database_path, "faiss_index.index"))
    
    user_query_embedding = utils.embedding_invoke(user_query, config)
    user_query_embedding = np.array(user_query_embedding).reshape(1, -1)

    D, I = faiss_index.search(user_query_embedding, k = config.embedding_top_k)

    return I, D

def RRF(bm25_result, embedding_result, ):
    bm25_indices, _ = bm25_result
    embedding_indices, _ = embedding_result

    fusion_scores = {}
    for rank, index in enumerate(bm25_indices):
        if index not in fusion_scores:
            fusion_scores[index] = 0
        fusion_scores[index] += 1 / (60 + rank + 1)

    for rank, index in enumerate(embedding_indices[0]):
        if index not in fusion_scores:
            fusion_scores[index] = 0
        fusion_scores[index] += 1 / (60 + rank + 1)

    sorted_fusion_scores = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_fusion_scores

def rewrite_query(messages : list, current_message : str, config):
    
    new_message = utils.llm_invoke(prompt.rewrite_prompt_v1.format(context = str(messages), question = current_message), config)
    return new_message

def ReRanker_search(user_query, database_content, fusion_scores, config : utils.Config):
    rank_score  = {fusion_scores[i][0].item(): 0 for i in range(config.rrf_top_k)}
    reranker = FlagReranker(config.reranker_path, use_fp16=True)
    for i, (index, score) in enumerate(fusion_scores[: config.rrf_top_k]):

        if index not in rank_score:
            assert(f"{index} not in rank_score dict")

        rank_score[index] = reranker.compute_score(
                                [
                                    user_query,
                                    list(database_content[index].values())[0]
                                ]
                            )
        
    return sorted(rank_score.items(), key=lambda x: x[1], reverse=True)



def main():
    config = utils.Config()
    messages = []
    database_index, database_content = init_database(config)

    while True:
        current_message = input("query : ")
        if current_message == "exit": break
        elif current_message == "clean":
            messages = []
            print("历史已经清空")
            continue
        

        user_query = rewrite_query(messages, current_message, config)
        print(user_query)
        

        bm25_result = BM25_search(user_query, database_content, config)
        embedding_result = embedding_search(user_query, database_index, config)
        # print("BM25 Result:", bm25_result)
        # print("Embedding Result:", embedding_result)
        fusion_scores = RRF(bm25_result, embedding_result)
        rerank_result = ReRanker_search(user_query, database_content, fusion_scores, config)

        content = ""
        for i, (index, score) in enumerate(rerank_result[: config.reranker_top_k]):
            information = list(database_content[index].keys())[0]
            _index = information.rfind("_")
            content += f"###title : {information[:_index]} \n###number : {information[_index + 1:]}\n"
            content += "content : " + database_content[index].get(information) + "\n\n"

        # 第一种加入历史的方式，是将指令，上下文，提问和回答都加入进去。但这会导致上下文过长且会干扰模型的输出
        # messages.append({"role" : "user", "content" : prompt.prompt_v3.format(context="\n".join(content), question=user_query)})

        # 第二种加入历史的方式，是只加入提问和模型的回答
        messages.append({"role" : "user", "content" : user_query})

        answear = utils.llm_invoke(prompt.prompt_v3.format(context=content, question=user_query), config)
        print(content)
        print(answear)
        messages.append({"role" : "assistant", "content" : answear})
if __name__ == "__main__":
    main()
    # print(np.array([1]).data())
