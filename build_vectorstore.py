import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_community.document_loaders import DocxLoader
except ImportError:
    from langchain_community.document_loaders import Docx2txtLoader as DocxLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
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

documents = []
for file_path in sorted(DATA_DIR.iterdir()):
    if file_path.suffix.lower() == '.pdf':
        documents.extend(PyPDFLoader(str(file_path)).load())
    elif file_path.suffix.lower() == '.txt':
        documents.extend(TextLoader(str(file_path), encoding='utf-8').load())
    elif file_path.suffix.lower() == '.docx':
        documents.extend(DocxLoader(str(file_path)).load())

if not documents:
    raise FileNotFoundError(f'No PDF, TXT, or DOCX files found in {DATA_DIR}')

print(f'Loaded {len(documents)} source sections')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
)
chunks = splitter.split_documents(documents)
print(f'Split into {len(chunks)} text chunks')

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(DB_DIR),
)
print(f'Local vector store saved to {DB_DIR}')
