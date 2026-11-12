import html
import os
import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_community.document_loaders import DocxLoader
except ImportError:
    from langchain_community.document_loaders import Docx2txtLoader as DocxLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
DB_DIR = PROJECT_DIR / 'vectorstore'
MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
DATA_DIR.mkdir(exist_ok=True)

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

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

prompt = ChatPromptTemplate.from_template('''
\u4f60\u662f\u77e5\u8bc6\u5e93\u95ee\u7b54\u52a9\u624b\u3002\u53ea\u80fd\u6839\u636e\u4e0b\u9762\u7684\u53c2\u8003\u8d44\u6599\u56de\u7b54\uff1b\u8d44\u6599\u91cc\u6ca1\u6709\u5c31\u660e\u786e\u8bf4\u201c\u8d44\u6599\u4e2d\u672a\u63d0\u53ca\u201d\uff0c\u4e0d\u8981\u7f16\u9020\u3002

\u53c2\u8003\u8d44\u6599\uff1a
{context}

\u5bf9\u8bdd\u5386\u53f2\uff1a
{history}

\u95ee\u9898\uff1a{question}
''')

chain = prompt | llm | StrOutputParser()

PROCESSING_TEXT = '\u5904\u7406\u4e2d\uff0c\u8bf7\u7a0d\u5019\u2026'
GENERATING_TEXT = '\u6b63\u5728\u751f\u6210\u2026'
MAX_DOCUMENT_ROWS = 20


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def format_history(history):
    lines = []
    for message in history or []:
        role = message.get('role', '') if isinstance(message, dict) else ''
        content = message.get('content', '') if isinstance(message, dict) else ''
        if isinstance(content, list):
            content = ''.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )
        if content:
            label = '\u7528\u6237' if role == 'user' else '\u52a9\u624b'
            lines.append(f'{label}\uff1a{content}')
    return '\n'.join(lines[-8:])


def collection_data():
    return vectorstore.get(include=['metadatas'])


def document_records():
    data = collection_data()
    records = {}
    for metadata in data.get('metadatas') or []:
        if not metadata or not metadata.get('source'):
            continue
        filename = Path(str(metadata['source'])).name
        record = records.setdefault(filename, {'filename': filename, 'chunks': 0})
        record['chunks'] += 1
    return sorted(records.values(), key=lambda item: item['filename'].lower())


def document_stats(records=None):
    records = document_records() if records is None else records
    total_chunks = sum(item['chunks'] for item in records)
    return f'\u6587\u6863 **{len(records)}** \u4efd  \u00b7  \u6587\u672c\u5757 **{total_chunks}** \u4e2a'


def document_view(status=''):
    records = document_records()
    result = [status, document_stats(records)]
    for slot in range(MAX_DOCUMENT_ROWS):
        if slot < len(records):
            record = records[slot]
            result.extend([html.escape(record['filename']), str(record['chunks']) + ' \u4e2a\u6587\u672c\u5757', gr.update(visible=True)])
        else:
            result.extend(['', '', gr.update(visible=False)])
    return result


def vector_ids_for_filename(filename):
    data = collection_data()
    return [
        doc_id
        for doc_id, metadata in zip(data.get('ids') or [], data.get('metadatas') or [])
        if metadata and metadata.get('source')
        and Path(str(metadata['source'])).name == filename
    ]


def remove_document_vectors(filename):
    ids = vector_ids_for_filename(filename)
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def ingest(file):
    if file is None:
        return document_view('\u8bf7\u5148\u9009\u62e9\u6587\u4ef6')

    source_path = Path(file if isinstance(file, (str, os.PathLike)) else file.name)
    filename = source_path.name
    extension = source_path.suffix.lower()
    if extension not in {'.pdf', '.txt', '.docx'}:
        return document_view('\u5165\u5e93\u5931\u8d25\uff1a\u4ec5\u652f\u6301 PDF / TXT / DOCX \u683c\u5f0f')

    save_path = DATA_DIR / filename
    copied_new_file = False
    had_existing_file = save_path.exists()
    try:
        if source_path.resolve() != save_path.resolve():
            shutil.copy2(source_path, save_path)
            copied_new_file = True
        if extension == '.pdf':
            docs = PyPDFLoader(str(save_path)).load()
        elif extension == '.txt':
            docs = TextLoader(str(save_path), encoding='utf-8').load()
        else:
            docs = DocxLoader(str(save_path)).load()
        chunks = splitter.split_documents(docs)
        remove_document_vectors(filename)
        vectorstore.add_documents(chunks)
        return document_view(
            f'\u5df2\u5165\u5e93\uff1a{filename}\uff08{len(chunks)} \u4e2a\u6587\u672c\u5757\uff09'
        )
    except Exception as exc:
        if copied_new_file and not had_existing_file and save_path.exists():
            save_path.unlink()
        return document_view(f'\u5165\u5e93\u5931\u8d25\uff1a{type(exc).__name__}: {exc}')


def delete_document(slot):
    records = document_records()
    if slot >= len(records):
        return document_view('\u5220\u9664\u5931\u8d25\uff1a\u6587\u6863\u4e0d\u5b58\u5728')
    filename = records[slot]['filename']
    try:
        chunk_count = remove_document_vectors(filename)
        file_path = DATA_DIR / filename
        if file_path.exists():
            file_path.unlink()
        return document_view(f'\u5df2\u5220\u9664\uff1a{filename}\uff08{chunk_count} \u4e2a\u6587\u672c\u5757\uff09')
    except Exception as exc:
        return document_view(f'\u5220\u9664\u5931\u8d25\uff1a{type(exc).__name__}: {exc}')


def clear_knowledge_base():
    try:
        data = collection_data()
        ids = data.get('ids') or []
        if ids:
            vectorstore.delete(ids=ids)
        removed_files = 0
        for file_path in DATA_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.txt', '.docx'}:
                file_path.unlink()
                removed_files += 1
        return document_view(
            f'\u5df2\u6e05\u7a7a\u77e5\u8bc6\u5e93\uff1a\u5220\u9664 {removed_files} \u4efd\u6587\u6863\uff0c\u79fb\u9664 {len(ids)} \u4e2a\u6587\u672c\u5757'
        ) + [gr.update(value=False), gr.update(interactive=False)]
    except Exception as exc:
        return document_view(f'\u6e05\u7a7a\u5931\u8d25\uff1a{type(exc).__name__}: {exc}') + [
            gr.update(value=True), gr.update(interactive=True)
        ]


def legacy_format_sources(docs):
    if not docs:
        return ''
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = html.escape(Path(str(doc.metadata.get('source', '\u672a\u77e5\u6765\u6e90'))).name)
        page = doc.metadata.get('page')
        page_label = f' \uff08\u7b2c {page + 1} \u9875\uff09' if isinstance(page, int) else ''
        snippet = html.escape(doc.page_content.strip()).replace('\n', '<br>')
        blocks.append(
            f'<div class=source-item><strong>{index}. {source}{page_label}</strong>'
            f'<blockquote>{snippet}</blockquote></div>'
        )
    return (
        f'<details class=source-reference><summary>\u53c2\u8003\u6765\u6e90（{len(docs)}）</summary>'
        + ''.join(blocks) + '</details>'
    )


def format_sources(docs):
    if not docs:
        return ''
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = html.escape(Path(str(doc.metadata.get('source', 'unknown source'))).name)
        page = doc.metadata.get('page')
        page_label = f' (page {page + 1})' if isinstance(page, int) else ''
        snippet = html.escape(doc.page_content.strip()).replace('\n', '<br>')
        blocks.append(
            f'<div class=source-item><strong>{index}. {source}{page_label}</strong>'
            f'<blockquote>{snippet}</blockquote></div>'
        )
    return (
        f'<details class=source-reference><summary>\u53c2\u8003\u6765\u6e90 ({len(docs)})</summary>'
        + ''.join(blocks) + '</details>'
    )


def answer(question, history):
    history = list(history or [])
    if not question or not question.strip():
        return history, ''
    question = question.strip()
    history_text = format_history(history)
    try:
        if not (collection_data().get('ids') or []):
            response = '\u77e5\u8bc6\u5e93\u4e3a\u7a7a\uff0c\u8bf7\u5148\u4e0a\u4f20 PDF / TXT / DOCX \u6587\u6863\u3002'
        else:
            retrieval_query = question
            if history_text:
                retrieval_query = f'{history_text}\n\u5f53\u524d\u95ee\u9898\uff1a{question}'
            docs = retriever.invoke(retrieval_query)
            if docs:
                response = chain.invoke({
                    'context': format_docs(docs),
                    'history': history_text or '\u65e0',
                    'question': question,
                })
                response = f'{response}{format_sources(docs)}'
            else:
                response = '\u8d44\u6599\u4e2d\u672a\u63d0\u53ca\u8be5\u95ee\u9898\u3002'
    except Exception as exc:
        response = f'\u8c03\u7528\u5931\u8d25\uff1a\u6682\u65f6\u65e0\u6cd5\u5b8c\u6210\u56de\u7b54\uff0c\u8bf7\u68c0\u67e5 API \u914d\u7f6e\u6216\u7f51\u7edc\u8fde\u63a5\u3002\n\n\u9519\u8bef\u7c7b\u578b\uff1a{type(exc).__name__}'
    history.extend([
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': response},
    ])
    return history, ''


def set_question_button(value):
    return gr.update(interactive=bool(value and value.strip()))


def start_upload():
    return PROCESSING_TEXT, gr.update(interactive=False)


def finish_upload():
    return gr.update(interactive=True)


def start_delete():
    return PROCESSING_TEXT, gr.update(interactive=False)


def finish_delete():
    return gr.update(interactive=True)


def start_clear():
    return PROCESSING_TEXT, gr.update(interactive=False)


def start_answer():
    return GENERATING_TEXT, gr.update(value='\u751f\u6210\u4e2d\u2026', interactive=False)


def finish_answer():
    return '', gr.update(value='\u63d0\u95ee', interactive=False)


def clear_chat():
    return []


CSS = '''
:root { --app-bg:#07111f; --panel-bg:#0d1b2e; --panel-border:#203957; --text-main:#e6f0ff; --text-muted:#91a7c4; --accent:#2f80ed; }
.gradio-container { max-width:1280px !important; background:var(--app-bg) !important; color:var(--text-main) !important; }
#app-header { border-bottom:1px solid var(--panel-border); padding:20px 8px 16px; margin-bottom:18px; }
#app-title h1 { color:#f4f8ff; letter-spacing:0; margin-bottom:8px; }
#app-subtitle { color:var(--text-muted); font-size:15px; }
.app-panel { background:var(--panel-bg) !important; border:1px solid var(--panel-border) !important; border-radius:8px !important; padding:18px !important; }
.panel-title h2 { color:#cfe2ff; font-size:18px; margin-top:0; }
.document-stats { color:#b9d5f7; background:#102945; border:1px solid #214568; border-radius:6px; padding:10px 12px; margin:10px 0 14px; }
.document-row { align-items:center; border-bottom:1px solid #1d3552; gap:8px; padding:9px 0; }
.document-name { color:#e6f0ff; overflow-wrap:anywhere; }
.document-chunks { color:var(--text-muted); font-size:12px; }
.document-delete button { min-width:54px; padding:4px 8px; font-size:12px; }
.knowledge-clear { border-top:1px solid var(--panel-border); margin-top:18px; padding-top:14px; }
#chatbot { border:1px solid var(--panel-border); border-radius:8px; background:#09172a; }
.source-reference { border-top:1px solid #29415d; color:#9eb4d0; font-size:12px; margin-top:12px; padding-top:8px; }
.source-reference summary { color:#79b4ff; cursor:pointer; font-weight:600; }
.source-item { border-bottom:1px solid #213b58; margin-top:8px; padding:6px 0 10px; }
.source-item blockquote { border-left:2px solid #367fc8; color:#afc2da; margin:6px 0 0; padding-left:9px; }
button.primary { background:var(--accent) !important; border-color:var(--accent) !important; }
#app-footer { border-top:1px solid var(--panel-border); color:var(--text-muted); font-size:13px; margin-top:18px; padding:14px 8px 4px; }
.document-stats { color:#f4f8ff; font-size:16px; font-weight:700; letter-spacing:.2px; background:#17395f; border-color:#3e79b7; }
.document-row { align-items:center !important; display:flex !important; flex-wrap:nowrap !important; min-height:42px; }
.document-row .document-name { flex:1 1 auto !important; min-width:0 !important; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.document-row .document-chunks { flex:0 0 88px !important; min-width:88px; text-align:right; white-space:nowrap; align-self:center; }
.document-row .document-delete { flex:0 0 64px !important; min-width:64px; align-self:center; }
.question-row { align-items:stretch !important; gap:10px; }
.question-row textarea { min-height:48px !important; }
.question-row button { min-height:48px; min-width:112px; }
.example-questions { color:#91a7c4; font-size:12px; margin:6px 0 12px; }
.qa-header { align-items:center !important; justify-content:space-between; }
.qa-header .panel-title { flex:1 1 auto; }
.clear-chat button { border-color:#315276; color:#b9d5f7; }
.top-nav { align-items:center; display:flex; justify-content:space-between; padding:2px 0 18px; }
.brand-lockup { align-items:center; display:flex; gap:12px; }
.brand-mark { align-items:center; background:#2f80ed; border:1px solid #6eafff; border-radius:10px; box-shadow:0 8px 22px rgba(47,128,237,.24); color:#fff; display:flex; font-size:20px; font-weight:800; height:38px; justify-content:center; width:38px; }
.brand-name { color:#f4f8ff; font-size:17px; font-weight:700; letter-spacing:0; }
.brand-name span { color:#cfe2ff; font-weight:500; }
.brand-caption { color:#6f8eaf; font-size:10px; letter-spacing:1.2px; margin-top:3px; text-transform:uppercase; }
.tech-tags, .feature-tags { display:flex; flex-wrap:wrap; gap:8px; }
.tech-tag, .feature-tags span { border:1px solid #315276; border-radius:999px; color:#a8c9ee; font-size:11px; padding:5px 10px; }
.tech-tag { background:#102945; }
.hero-block { border-top:1px solid #1c3551; padding:26px 0 8px; }
.hero-kicker { color:#6aaeff; font-size:11px; font-weight:700; letter-spacing:1.4px; text-transform:uppercase; }
.hero-block h1 { color:#f5f9ff; font-size:38px; font-weight:750; line-height:1.2; margin:10px 0 10px; }
.hero-block p { color:#9db4d0; font-size:14px; line-height:1.7; margin:0 0 15px; }
.feature-tags span { background:#0d2139; color:#b9d5f7; }
.content-grid { gap:20px; }
.app-panel { box-shadow:0 16px 38px rgba(0,0,0,.18) !important; }
.section-heading { align-items:center; display:flex; gap:10px; margin-bottom:14px; }
.section-heading strong { align-items:center; background:#153b67; border:1px solid #3e79b7; border-radius:8px; color:#79b4ff; display:flex; font-size:16px; height:30px; justify-content:center; width:30px; }
.section-heading span { display:flex; flex-direction:column; gap:3px; }
.section-heading b { color:#dceaff; font-size:17px; font-weight:700; }
.section-heading small { color:#7794b4; font-size:11px; }
.list-heading { align-items:center; color:#9db9d8; display:flex; font-size:12px; font-weight:700; gap:10px; margin:20px 0 4px; }
.list-heading-line { background:#284866; flex:1; height:1px; }
.document-stats { box-shadow:inset 0 0 0 1px rgba(121,180,255,.08); }
.app-panel input:focus, .app-panel textarea:focus, #question-input textarea:focus { border-color:#4d9cff !important; box-shadow:0 0 0 3px rgba(47,128,237,.18) !important; }
.app-panel button { transition:background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease; }
.app-panel button:hover:not([disabled]) { box-shadow:0 5px 14px rgba(0,0,0,.18); transform:translateY(-1px); }
.app-panel button.primary { background:#2f80ed !important; color:#fff !important; }
.app-panel button.primary:hover:not([disabled]) { background:#4d9cff !important; }
.app-panel button.stop { background:#7f3048 !important; border-color:#a84a64 !important; color:#fff !important; }
.app-panel button.stop:hover:not([disabled]) { background:#a23d59 !important; }
.document-delete button { background:#55283a !important; border-color:#8d3b56 !important; color:#ffdce5 !important; }
.document-delete button:hover:not([disabled]) { background:#7f3048 !important; border-color:#b6506b !important; }
.clear-chat button { background:#132943 !important; }
.clear-chat button:hover:not([disabled]) { background:#1b3b61 !important; }
#chatbot { box-shadow:inset 0 1px 0 rgba(255,255,255,.02); min-height:420px; }
#chatbot .message.user { background:#1c5ca9 !important; border:1px solid #397fce !important; color:#f7fbff !important; }
#chatbot .message.bot { background:#10243b !important; border:1px solid #294766 !important; color:#dceaff !important; }
#chatbot .message { border-radius:12px !important; box-shadow:0 5px 14px rgba(0,0,0,.12); }
.source-reference { background:#0a192b; border:1px solid #223f5d; border-radius:7px; padding:8px 10px; }
.source-reference summary { padding:2px 0; }
#app-footer { align-items:center; display:flex; gap:7px; justify-content:center; }
.footer-dot { background:#4d9cff; border-radius:50%; box-shadow:0 0 10px rgba(77,156,255,.8); display:inline-block; height:5px; width:5px; }
@media (max-width: 760px) {
    #app-header { padding:16px 2px 12px; }
    .top-nav { align-items:flex-start; flex-direction:column; gap:14px; }
    .tech-tags { gap:6px; }
    .hero-block h1 { font-size:27px; }
    .content-grid { flex-direction:column !important; }
    .content-grid > .app-panel { width:100% !important; }
    #chatbot { min-height:340px; }
    .document-row .document-chunks { flex-basis:78px !important; min-width:78px; }
}
'''

THEME = gr.themes.Soft(primary_hue='blue', secondary_hue='blue', neutral_hue='slate')
initial_records = document_records()

with gr.Blocks(title='DocuQA \u79c1\u6709\u77e5\u8bc6\u5e93\u52a9\u624b') as demo:
    with gr.Column(elem_id='app-header'):
        gr.HTML('<nav class=top-nav><div class=brand-lockup><div class=brand-mark>D</div><div><div class=brand-name>DocuQA <span>\u79c1\u6709\u77e5\u8bc6\u5e93\u52a9\u624b</span></div><div class=brand-caption>Private knowledge, clearly answered</div></div></div><div class=tech-tags><span class=tech-tag>RAG</span><span class=tech-tag>Chroma</span><span class=tech-tag>Local AI</span></div></nav>')
        gr.HTML('<section class=hero-block><div class=hero-kicker>\u79c1\u6709\u6587\u6863\u667a\u80fd\u95ee\u7b54</div><h1>\u8ba9\u6bcf\u4e00\u4efd\u6587\u6863\u90fd\u53d8\u5f97\u53ef\u68c0\u7d22\u3001\u53ef\u7406\u89e3</h1><p>\u57fa\u4e8e RAG \u68c0\u7d22\u589e\u5f3a\u751f\u6210\u6280\u672f\uff0c\u652f\u6301\u79c1\u6709\u6587\u6863\u4e0a\u4f20\u4e0e\u7cbe\u51c6\u6eaf\u6e90\u95ee\u7b54\u3002</p><div class=feature-tags><span>\u79c1\u6709\u6587\u6863\u4e0a\u4f20</span><span>RAG \u7cbe\u51c6\u68c0\u7d22</span><span>\u7b54\u6848\u6eaf\u6e90</span></div></section>')

    with gr.Row(equal_height=False, elem_classes='content-grid'):
        with gr.Column(scale=3, elem_classes='app-panel'):
            gr.HTML('<div class=section-heading><div class=section-icon>+</div><div><div class=section-title>\u6587\u6863\u7ba1\u7406</div><div class=section-subtitle>\u4e0a\u4f20\u3001\u7ef4\u62a4\u4e0e\u7ba1\u7406\u77e5\u8bc6\u5e93</div></div></div>')
            stats = gr.Markdown(document_stats(initial_records), elem_classes='document-stats')
            file = gr.File(label='\u4e0a\u4f20 PDF / TXT / DOCX \u6587\u6863', file_types=['.pdf', '.txt', '.docx'], type='filepath')
            upload_btn = gr.Button('\u4e0a\u4f20\u5e76\u5165\u5e93', variant='primary')
            upload_status = gr.Markdown()
            gr.HTML('<div class=list-heading><span>\u5df2\u5165\u5e93\u6587\u6863</span><span class=list-heading-line></span></div>')

            document_rows, delete_buttons, document_names, document_chunks = [], [], [], []
            for slot in range(MAX_DOCUMENT_ROWS):
                record = initial_records[slot] if slot < len(initial_records) else None
                with gr.Row(visible=record is not None, elem_classes='document-row') as document_row:
                    name = gr.HTML(html.escape(record['filename']) if record else '', elem_classes='document-name', scale=5)
                    chunks = gr.HTML((str(record['chunks']) + ' \u4e2a\u6587\u672c\u5757') if record else '', elem_classes='document-chunks', scale=3)
                    delete_btn = gr.Button('\u5220\u9664', variant='stop', size='sm', elem_classes='document-delete', scale=0)
                document_rows.append(document_row)
                delete_buttons.append(delete_btn)
                document_names.append(name)
                document_chunks.append(chunks)

            clear_confirm = gr.Checkbox(
                label='\u6211\u786e\u8ba4\u6e05\u7a7a\u5168\u90e8\u77e5\u8bc6\u5e93\u5185\u5bb9',
                value=False,
                elem_classes='knowledge-clear',
            )
            clear_btn = gr.Button('\u6e05\u7a7a\u77e5\u8bc6\u5e93', variant='stop', interactive=False)

        with gr.Column(scale=7, elem_classes='app-panel'):
            with gr.Row(elem_classes='qa-header'):
                gr.HTML('<div class=section-heading><strong>Q</strong><span><b>\u77e5\u8bc6\u5e93\u95ee\u7b54</b><small>\u57fa\u4e8e\u6587\u6863\u5185\u5bb9\u7684\u591a\u8f6e\u5bf9\u8bdd</small></span></div>')
                clear_chat_btn = gr.Button('\u6e05\u7a7a\u5bf9\u8bdd', variant='secondary', size='sm', elem_classes='clear-chat')
            chatbot = gr.Chatbot(
                value=[],
                show_label=False,
                height=520,
                layout='bubble',
                elem_id='chatbot',
                placeholder='\u4e0a\u4f20\u6587\u6863\u540e\uff0c\u5728\u4e0b\u65b9\u8f93\u5165\u95ee\u9898\u5f00\u59cb\u5bf9\u8bdd',
            )
            with gr.Row(elem_classes='question-row'):
                question = gr.Textbox(
                    label='\u63d0\u95ee',
                    placeholder='\u8f93\u5165\u95ee\u9898\uff0c\u6309 Enter \u63d0\u4ea4',
                    lines=2,
                    scale=8,
                    elem_id='question-input',
                )
                answer_btn = gr.Button('\u63d0\u95ee', variant='primary', interactive=False, scale=0, min_width=112)
            examples = gr.Examples(
                examples=[
                    ['\u8fdb\u653b\u65b9\u8981\u505a\u4ec0\u4e48\uff1f'],
                    ['\u8fd9\u4efd\u8d44\u6599\u7684\u5173\u952e\u6b65\u9aa4\u662f\u4ec0\u4e48\uff1f'],
                    ['\u4e0a\u4f20\u6587\u6863\u540e\u5982\u4f55\u67e5\u8be2\u5176\u5185\u5bb9\uff1f'],
                ],
                inputs=question,
                label='\u8bd5\u8bd5\u8fd9\u4e9b\u95ee\u9898',
                examples_per_page=3,
            )
            answer_status = gr.Markdown()

    gr.HTML('<footer id=app-footer><span class=footer-dot></span>Python <b>\u00b7</b> LangChain <b>\u00b7</b> Chroma <b>\u00b7</b> Gradio <b>\u00b7</b> \u672c\u5730\u5411\u91cf\u5316</footer>')

    document_outputs = [upload_status, stats]
    for name, chunks, row in zip(document_names, document_chunks, document_rows):
        document_outputs.extend([name, chunks, row])

    upload_event = upload_btn.click(
        start_upload,
        outputs=[upload_status, upload_btn],
        show_progress='hidden',
    )
    upload_event.then(
        ingest,
        inputs=file,
        outputs=document_outputs,
        show_progress='hidden',
    ).then(finish_upload, outputs=upload_btn, show_progress='hidden')

    for slot, delete_btn in enumerate(delete_buttons):
        delete_event = delete_btn.click(
            start_delete,
            outputs=[upload_status, delete_btn],
            show_progress='hidden',
        )
        delete_event.then(
            lambda slot=slot: delete_document(slot),
            outputs=document_outputs,
            show_progress='hidden',
        ).then(finish_delete, outputs=delete_btn, show_progress='hidden')

    clear_confirm.change(
        lambda confirmed: gr.update(interactive=bool(confirmed)),
        inputs=clear_confirm,
        outputs=clear_btn,
        show_progress='hidden',
    )
    clear_event = clear_btn.click(
        start_clear,
        outputs=[upload_status, clear_btn],
        show_progress='hidden',
    )
    clear_event.then(
        clear_knowledge_base,
        outputs=document_outputs + [clear_confirm, clear_btn],
        show_progress='hidden',
    )

    question.input(
        set_question_button,
        inputs=question,
        outputs=answer_btn,
        show_progress='hidden',
    )

    def answer_event_chain(trigger):
        event = trigger(
            start_answer,
            outputs=[answer_status, answer_btn],
            show_progress='hidden',
        )
        event = event.then(
            answer,
            inputs=[question, chatbot],
            outputs=[chatbot, question],
            show_progress='hidden',
        )
        event.then(
            finish_answer,
            outputs=[answer_status, answer_btn],
            show_progress='hidden',
        )

    answer_event_chain(answer_btn.click)
    answer_event_chain(question.submit)
    clear_chat_btn.click(
        clear_chat,
        outputs=chatbot,
        show_progress='hidden',
    )


if __name__ == '__main__':
    demo.queue().launch(
        server_name='127.0.0.1',
        server_port=7860,
        theme=THEME,
        css=CSS,
    )

