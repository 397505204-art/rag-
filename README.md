# DocuQA 私有知识库问答助手

一个基于 Python + LangChain + Chroma + 本地 Embedding + Gradio 的 RAG 私有知识库问答助手。

项目支持将 PDF、TXT、DOCX 文档切片后使用本地 `BAAI/bge-small-zh-v1.5` 模型向量化，保存到本地 Chroma 向量库，并通过 OpenAI 兼容的大模型 API 完成检索增强问答。回答会附带实际命中的文件、页码和原文片段，便于核验。

## 技术栈

- Python 3.11+
- LangChain：文档加载、文本切片、检索和问答链
- Chroma：本地持久化向量数据库
- Sentence Transformers：本地中文 Embedding 模型
- Gradio：深色科技风网页界面
- OpenAI 兼容 API：大模型回答生成

## 功能列表

- 支持 PDF / TXT / DOCX 文档上传、切片和向量化入库
- 支持多轮对话与上下文追问
- 回答附带可折叠的参考来源和原文片段
- 支持查看文档数、文本块数和已入库文档列表
- 支持删除单篇文档，并同步删除对应向量
- 支持二次确认后清空整个知识库
- 支持 API 失败、格式不支持、知识库为空等异常提示
- 支持响应式左右分栏界面和示例问题

## 环境依赖

建议使用 Python 3.11 或更高版本。依赖版本见 [requirements.txt](requirements.txt)。首次运行会下载本地 Embedding 模型，模型缓存默认位于项目的 `models/` 目录。

## 本地启动

1. 创建虚拟环境并安装依赖：

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. 复制 `.env.example` 为 `.env`，填写自己的 API 配置：

   ```ini
   API_KEY=你的 API 密钥
   BASE_URL=https://你的 OpenAI 兼容接口地址/v1
   CHAT_MODEL=你的对话模型名
   EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
   ```

   `.env` 只用于本地配置，禁止提交到公开仓库。

3. 启动 Gradio 应用：

   ```powershell
   python app.py
   ```

4. 浏览器访问 `http://127.0.0.1:7860`。

如需批量重建向量库，可将 PDF、TXT 或 DOCX 放入 `data/` 后运行：

```powershell
python build_vectorstore.py
```

## 目录结构

```text
rag-assistant/
├── app.py                 # Gradio 应用入口
├── build_vectorstore.py   # 批量加载文档并构建向量库
├── qa.py                  # 命令行检索问答
├── test_api.py            # API 连通性测试
├── requirements.txt       # 固定版本依赖
├── .env.example           # 环境变量模板
├── data/                  # 本地私有文档，不提交
├── models/                # 本地 Embedding 缓存，不提交
└── vectorstore/           # Chroma 持久化目录，不提交
```

## 界面截图

截图占位：运行 `python app.py` 后，打开 `http://127.0.0.1:7860` 截取完整界面，并将图片保存为 `docs/screenshot.png`，再替换下面的占位路径：

`![DocuQA 界面截图](docs/screenshot.png)`

## 安全说明

- `.env`、API 密钥、私有文档、向量库和本地模型缓存均已加入 `.gitignore`。
- 发布到公开仓库前，请确认没有把真实密钥写入源码、README、Notebook 或日志。
- 如果密钥曾经被公开，应立即在服务商后台撤销并重新生成。
