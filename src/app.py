import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. 配置你的 API（⚠️ 第12行换成真实Key）
# ==========================================
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
CHAT_MODEL = "glm-4-flash"
EMBEDDING_MODEL = "embedding-3"

@st.cache_resource
def init_rag_chain():
    # 加载文档
    loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    
    # 切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_documents(docs)
    
    # 向量化
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 提示词（已加上引用来源指令）
    system_prompt = """你是一个客服助手。请严格根据以下参考材料回答用户问题。
如果参考材料中没有相关信息，请直接回答：“抱歉，我暂时无法确认。我可以帮您转接人工客服，请稍等。”
回答后，必须在最后另起一行，写上：“📚 来源：”加上参考材料中的来源文件名。

参考材料：
{context}
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    
    # 模型
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.1)
    
    # 组装链条（已加上带来源的格式化函数）
    def format_docs(docs):
        return "\n\n".join(f"【来源：{doc.metadata.get('source', '未知文档')}】\n{doc.page_content}" for doc in docs)
        
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# ==========================================
# 2. 初始化网页界面
# ==========================================
st.set_page_config(page_title="校园数码小店客服", page_icon="🎧")
st.title("🎧 校园数码小店客服助手")

# 显示聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是校园数码小店的客服助手，请问有什么可以帮您？"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("客服正在查询知识库..."):
        rag_chain = init_rag_chain()
        answer = rag_chain.invoke(prompt)
        
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)