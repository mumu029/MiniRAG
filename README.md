# MiniRAG
一个非常简单的，用于入门学习的RAG系统。实现了基本的RAG功能，且代码非常简单。具体的功能包括：
1. 基于语义的向量检索
2. 基于字符的BM25词频检索
3. 基于RRF的多路混合检索
4. 基于智源模型的重排
5. 基于历史的用户输入重写


# 环境准备
不需要复杂的环境，大部分都是常见的库。可能需要注意的有：
```shell
pip install rank_bm25 jieba FlagEmbedding faiss-cpu 
```
此外还有一个ReRanker模型需要下载：https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3

然后设置好自己的知识库目录地址，大模型的API Key和Embedding模型（建议使用智谱的Embedding模型）的API Key就可以了。

# 执行
```shell
python main.py
```
