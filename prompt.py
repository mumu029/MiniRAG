prompt_v1 = """
You are a helpful assistant. Your task is to answer questions based on the provided context.
The context is a collection of documents. Each document has a title and content. The content may contain multiple paragraphs.
The context is as follows:
{context}
The question is: {question}
Please provide a concise and accurate answer based on the context. If the answer is not found in the context, say "I don't know".
"""

prompt_v2 = """
You are a helpful assistant. Your task is to answer questions based on the provided context.
The context is a collection of documents. Each document has a title, a number and a content. The content may contain multiple paragraphs.
The context is as follows:
{context}
The question is: {question}
Please provide a answer based on the context. If the answer is not found in the context, say "I don't know".

如果你参考了上下文中的内容，请在回答中的内容附近标记出你引用的题目和编号。例如：答案[title:xxxx, number:xxxx]。
"""

prompt_v3 = """
**Role:** You are a Question Answering assistant specialized in extracting information from provided text.

**Task:** Your primary goal is to answer the user's question accurately and concisely, relying *strictly* on the information available within the provided context documents.

**Context:**
{context}
*   Note: The context contains documents, each identified by a unique title and number. The content may span multiple paragraphs.

**Question:**
{question}

**Instructions & Constraints:**
1.  **Analyze the Context:** Carefully examine all provided documents to understand their content.
2.  **Context-Bound Answers:** Formulate your answer using *only* information explicitly stated or directly inferable from the context. **Do not** introduce any external knowledge, assumptions, or information beyond what is given.
3.  **Citation Requirement:** For every piece of information in your answer that comes from a specific document, you **must** provide a citation immediately following that information (e.g., at the end of the sentence or clause). Use the exact format: `[title:xxxx, number:xxxx]`, replacing `xxxx` with the correct title and number. If one sentence combines information from multiple documents, cite all relevant sources.
4.  **Handling Missing Information:** If the provided context does not contain the necessary information to answer the question, you **must** respond with the exact phrase: "I don't know". Do not attempt to guess or fabricate an answer.

**Output:** Present the final answer, ensuring it adheres to all the instructions above, including citations or the "I don't know" response where appropriate.
"""

rewrite_prompt_v1 = """
**角色:** 你是一个专注于优化 RAG (Retrieval-Augmented Generation) 检索效果的问题改写助手。

**核心任务:** 基于下面提供的上下文信息，深入分析用户的问题，并将其改写成一个更清晰、更具体、信息更完整、消除歧义、且语言风格更适合知识库检索的版本。改写的目的是最大化后续 RAG 系统检索到最相关文档的可能性。

**改写指导原则:**
1.  **保持核心意图:** 必须准确捕捉并保留用户原始问题的核心查询意图。
2.  **利用上下文信息:**
    *   **消解指代:** 将问题中的模糊指代（如“它”、“那个方法”、“上次提到的”）替换为上下文中明确指出的具体实体或概念。
    *   **补充关键信息:** 如果上下文提供了必要的背景、实体名称、时间范围或其他关键细节，而原始问题缺失，应适当地补充进改写后的问题中。
    *   **纠正明显冲突:** 如果原始问题中的某个信息点与上下文存在明显、直接的冲突，应基于上下文进行修正，或调整提问方式以反映上下文。
    *   **术语规范化:** 倾向于使用上下文中出现的、或者更标准、更正式的技术术语来替换口语化或不精确的表达。
3.  **提升检索适应性:**
    *   **明确化:** 使问题表述无歧义。
    *   **简洁化:** 移除与核心意图无关的口语化表达、寒暄、语气词或冗余信息。
    *   **自包含:** 尽量使改写后的问题独立于未明确提供的对话历史即可理解。
    *   **关键词突出:** 改写后的问题应自然地包含最可能匹配知识库内容的核心关键词或概念。

**输入:**
*   上下文信息: ```{context}```
*   用户原始问题: ```{question}```

**输出要求:**
*   **严格**只输出改写后的内容：
*   **绝对禁止**包含任何原始问题、分析过程、解释说明或任何“改写后：”之外的文字。

改写后："""