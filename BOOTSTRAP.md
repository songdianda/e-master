# E-master：RAG 投资知识库对话系统

> "E大" 是知名指数投资者 ETF拯救世界的昵称。本项目汇集了他的公开文章、微博、长赢投资计划等思想资产，构建了一个本地 RAG 检索增强生成对话系统。

## 一句话概述

**上传博主的 600+ 篇文章 → 自动清洗分类 → 本地检索 + LLM 流式推理 → 像和博主本人对话一样获取投资与生活智慧。**

---

## 项目架构

```
E-master/
├── app.py                    # Streamlit RAG 对话控制台（核心应用）
├── clean_data.py             # 自动清洗引擎（txt/md/pdf/html → JSON + CSV）
├── requirements.txt          # Python 依赖
├── processed_articles.json   # RAG 数据底座（640 篇纯文本 + 标签）
├── 文章资产总台账.csv         # Excel 可视化台账
├── .gitignore                # 排除 raw_articles/、.rag_config.json 等
├── .streamlit/
│   └── config.toml           # Streamlit 配置
└── raw_articles/             # 原始文章（不提交 Git）
    ├── *.txt                 # 308 篇纯文本
    ├── *.md                  # 267 篇 Markdown
    ├── *.html                # 69 篇 HTML
    └── *.pdf.pdf.txt         # 26 篇 PDF 转文本
```

## 核心功能

### 1. 自动数据清洗 (`clean_data.py`)

- 万能读取：`.txt` / `.md` / `.pdf`(pypdf) / `.html`(BeautifulSoup)
- 智能分类：591 篇投资逻辑 / 34 篇生活感悟 / 11 篇个人成长
- 双重输出：`processed_articles.json`（RAG 底座）+ CSV 台账

### 2. RAG 对话控制台 (`app.py`)

- **本地检索**：Bigram 中文关键词提取 + 文本重叠打分
- **云端推理**：OpenAI 兼容 API 流式输出（支持 DeepSeek / Qwen / Kimi 等 7 个提供商）
- **三阶段模式**：硬核投资（理性分析）/ 温暖生活（焦虑疏导）/ 默认模式
- **追问改写**：LLM 自动将省略追问提炼为独立检索词
- **原文出处**：每个回复可展开查看引用文章标题、分类和摘录
- **云端部署**：已适配 Streamlit Cloud + `st.secrets` 安全密钥管理

### 3. 数据运维

```bash
# 增量更新文章
python clean_data.py
# 推送至云端
git add processed_articles.json 文章资产总台账.csv
git commit -m "data: 更新文章"
git push
```

---

## 一键启动

### 本地运行

```bash
cd E-master
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 云端地址

部署于 Streamlit Cloud：`https://songdianda-e-master.streamlit.app`

云端密钥通过 `st.secrets` 托管，访客侧边栏不显示 API 配置。

---

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Streamlit 1.50 |
| 推理 | OpenAI 兼容 API |
| 检索 | 本地 Bigram 关键词匹配 |
| 文本处理 | pypdf + BeautifulSoup4 |
| 部署 | GitHub + Streamlit Cloud |
| 数据 | JSON + CSV (utf-8-sig) |

## Git 仓库

`https://github.com/songdianda/e-master`（公开仓库，API 密钥由 Streamlit Secrets 托管）

---

## 常见问题

**Q: 如何新增文章？**
把文件丢进 `raw_articles/`，运行 `python clean_data.py`，提交并推送。

**Q: 如何切换 LLM 提供商？**
本地运行：侧边栏下拉选择（DeepSeek / 阿里百炼 / Kimi 等）。云端部署：在 Streamlit Cloud Settings → Secrets 中修改。

**Q: 检索不到原文怎么办？**
系统会自动触发追问改写，将省略的问句展开后再检索。如果仍不行，尝试在"硬核投资"或"温暖生活"模式下提问以缩小检索范围。
