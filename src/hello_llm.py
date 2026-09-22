import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 配置你的 API（请替换成真实的 Key）
API_KEY = "4303c380e2f24a32b2442429d6aa7376.4M1GBD5xOiW8chay"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
CHAT_MODEL = "glm-4-flash"
EMBEDDING_MODEL = "embedding-3"

# 2. 加载 docs 文件夹里的全部知识库文档
print("正在加载知识库文档...")
from langchain_community.document_loaders import DirectoryLoader, TextLoader
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
# 如果你想把所有文档都加载进去，可以把上面的 loader 换成：
# from langchain_community.document_loaders import DirectoryLoader
# loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
# docs = loader.load()

# 3. 把长文档切碎（切片）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # 每片最多300字
    chunk_overlap=50,    # 相邻两片重叠50字
    length_function=len
)
chunks = text_splitter.split_documents(docs)
print(f"文档已切分为 {len(chunks)} 个片段")

# 4. 把切片转换为向量，并存入本地向量数据库
print("正在向量化并存入数据库...")
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL
)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 找最相关的2个片段

# 5. 定义 RAG 的 Prompt
system_prompt = """你是一个客服助手。请严格根据以下参考材料回答用户问题。
如果参考材料中没有相关信息，请直接回答：“抱歉，我暂时无法确认。我可以帮您转接人工客服，请稍等。”

参考材料：
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# 6. 初始化大模型
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1
)

# 7. 组装 RAG 链条
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. 启动对话循环
print("校园数码小店RAG客服助手已启动，输入 q 退出。")
while True:
    q = input("\n用户：").strip()
    if q.lower() in ["q", "quit", "exit"]:
        break
    if not q:
        continue
    answer = rag_chain.invoke(q)
    print("客服助手：", answer)