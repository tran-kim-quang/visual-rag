# memory_setup.py
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1) Text buffer để lưu lịch sử chat dạng text
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 2) Vector store (Chroma) để lưu embeddings mỗi message
embeddings = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder'"
)
chroma_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_conversations"
)
vector_memory = VectorStoreRetrieverMemory(
    vectorstore=chroma_store,
    memory_key="conv_vectors",
    return_source_documents=False
)

# Hàm gọi để persist (gọi khi shutdown hoặc định kỳ)
def persist_memory():
    chroma_store.persist()
