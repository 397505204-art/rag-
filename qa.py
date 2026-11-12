import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / 'vectorstore'
MODEL_NAME = 'BAAI/bge-small-zh-v1.5'

load_dotenv(PROJECT_DIR / '.env')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder=str(PROJECT_DIR / 'models'),
)

vectorstore = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

llm = ChatOpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('BASE_URL'),
    model=os.getenv('CHAT_MODEL'),
)

prompt = ChatPromptTemplate.from_template('''
你是知识库问答助手。只能根据下面的参考资料回答；资料里没有就明确说“资料中未提及”，不要编造。

参考资料：
{context}

问题：{question}
''')


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def run_once(question: str) -> None:
    print('\n回答：', chain.invoke(question))


if __name__ == '__main__':
    command_line_question = ' '.join(sys.argv[1:]).strip()
    if command_line_question:
        run_once(command_line_question)
    else:
        while True:
            question = input('\n问题（输入 exit 退出）：')
            if question.strip().lower() == 'exit':
                break
            run_once(question)
