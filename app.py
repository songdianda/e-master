#!/usr/bin/env python3
"""
RAG 知识库对话控制台 — 基于 Streamlit + WorkBuddy API
读取 processed_articles.json，本地检索 + 云端推理，流式输出。
"""

import json
import re
import streamlit as st
from pathlib import Path
from typing import Optional
from collections import Counter

# ──────────────────────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="专属 AI 思想导师",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 全局样式修复
st.markdown("""
<style>
button[kind="secondary"] {
    white-space: nowrap !important;
}
/* 隐藏底部 Streamlit 品牌栏和右上角菜单 */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "processed_articles.json"
CONFIG_FILE = BASE_DIR / ".rag_config.json"

# ──────────────────────────────────────────────────────────────
# 配置加载优先级：st.secrets > 本地持久化 > 硬编码默认值
# ──────────────────────────────────────────────────────────────

def load_secrets() -> dict:
    """从 st.secrets 加载云部署配置（兼容本地 .streamlit/secrets.toml）。"""
    secrets = {}
    try:
        secrets["api_key"] = st.secrets.get("api_key", "")
        secrets["base_url"] = st.secrets.get("base_url", "")
        secrets["model_name"] = st.secrets.get("model_name", "")
        secrets["provider"] = st.secrets.get("provider", "")
    except Exception:
        pass
    return {k: v for k, v in secrets.items() if v}  # 过滤空值


def merge_config() -> dict:
    """按优先级合并：secrets > 本地文件 > 默认值，返回最终配置。"""
    cloud = load_secrets()
    local = load_config()

    return {
        "provider": cloud.get("provider") or local.get("provider") or DEFAULT_PROVIDER,
        "api_key": cloud.get("api_key") or local.get("api_key") or "",
        "base_url": cloud.get("base_url") or local.get("base_url") or DEFAULT_BASE_URL,
        "model_name": cloud.get("model_name") or local.get("model_name") or DEFAULT_MODEL,
    }


def load_config() -> dict:
    """从本地加载持久化配置（.rag_config.json）。"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config: dict):
    """保存配置到本地 JSON 文件。"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────────────────────
# 常用 API 提供商预设
# ──────────────────────────────────────────────────────────────
API_PROVIDERS = {
    "🔧 自定义": {
        "base_url": "",
        "model": "",
        "desc": "手动填写 Base URL 和模型名",
    },
    "DeepSeek (推荐⭐)": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "desc": "便宜好用，中文理解力强，10元/百万token",
    },
    "阿里百炼 (Qwen)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
        "desc": "阿里通义千问旗舰模型",
    },
    "Moonshot (Kimi)": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "desc": "月之暗面，长文本处理",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "desc": "需科学上网，价格较高",
    },
    "硅基流动 (SiliconFlow)": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-ai/DeepSeek-V3",
        "desc": "国产代理，免翻墙调各种模型",
    },
    "智谱 (GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "desc": "智谱AI，国产大模型",
    },
}

DEFAULT_PROVIDER = "DeepSeek (推荐⭐)"
DEFAULT_BASE_URL = API_PROVIDERS[DEFAULT_PROVIDER]["base_url"]
DEFAULT_MODEL = API_PROVIDERS[DEFAULT_PROVIDER]["model"]

# 中文停用词（极简版）
STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 "
    "会 着 没有 看 自己 这 他 她 它 们 那 这个 那个 什么 怎么 哪 为什么 "
    "如果 因为 所以 但是 虽然 可以 还是 只是 然后 已经 还是 如何 以及 "
    "之 与 或 且 但 而 从 向 以 对 被 让 请 把 给 用 通过 为了 关于 "
    "对于 由于 按照 除了 根据 随着 经过 通过 应该 需要 可能 能够 可以 "
    "不过 但是 而且 然而 因此 所以 于是 接着 然后 最后 首先 其次 再次 "
    "另外 此外 同时 并且 当然 果然 居然 竟然 其实 似乎 好像 或许 或许 "
    "大概 大约 也许 一定 必须 绝对 永远 从来 一直 总是 经常 偶尔 有时 "
    "不仅 不止 不论 无论 不管 任何 每一个 每个 每天 每年 每次 真正 非常 "
    "特别 更加 更加 尤其 极其 十分 相当 比较 稍微 略微 一点 一些 很多 "
    "大部分 大多数 少数 少数人 部分 全部 所有 整个 整 某 某些 别的 其他 "
    "另外 别的 剩下 剩余 此外 此外 除此之外 除此以外 至于 至于说 还有 "
    "另有 除此之外 另外 另 另一个 其他 其余 其它 另外 此外 除此之外 "
    "那么 这样 那样 怎么 什么样 怎么样 这么 那么 多么 为什么 为何 怎么 "
    "能不能 能不能够 可不可以 是不是 对不对 好不好 行不行 有没有 可不可以 "
    "个 次 种 些 点 分钟 小时 天 年 月 周 日 前 后 左 右 里 外 内 中 间 "
    "啊 吧 呢 吗 嘛 呀 哦 嗯 哈 唉 哎 喂 呗 啦 咯 噢 哟".split()
)


# ──────────────────────────────────────────────────────────────
# 文本分块
# ──────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str, title: str, chunk_size: int = 500, overlap: int = 80
) -> list[dict]:
    """将长文本切分为重叠块，每块记录来源标题和位置。"""
    # 按自然段拆分
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\n(?=[^\n]{20,})", text) if p.strip()]

    chunks = []
    idx = 0
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append({
                "title": title,
                "chunk_index": idx,
                "text": para,
            })
            idx += 1
        else:
            # 长段落按字符滑动窗口切分
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunk_text = para[start:end]
                chunks.append({
                    "title": title,
                    "chunk_index": idx,
                    "text": chunk_text,
                })
                idx += 1
                start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────────────────────
# 关键词提取（中文 Bigram + 实词）
# ──────────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """从中文文本提取关键词：Bigram + 过滤停用词。"""
    # 过滤非中文字符，只取中文
    chinese_only = re.sub(r"[^\u4e00-\u9fff]", "", text)
    if not chinese_only:
        return []

    # Bigram 生成
    bigrams = [chinese_only[i:i + 2] for i in range(len(chinese_only) - 1)]
    # 过滤纯停用词 Bigram
    bigrams = [bg for bg in bigrams if not all(c in STOP_WORDS for c in bg)]

    counts = Counter(bigrams)
    # 取 top_n
    return [bg for bg, _ in counts.most_common(top_n)]


# ──────────────────────────────────────────────────────────────
# 查询改写（解决追问时检索失效问题）
# ──────────────────────────────────────────────────────────────

FOLLOWUP_SIGNALS = [
    "为什么", "为啥", "怎么", "如何", "这", "那", "它", "他", "她",
    "具体", "详细", "举例", "比如", "然后", "所以", "因此",
    "呢", "吗", "吧", "还有", "继续",
]


def needs_rewrite(query: str) -> bool:
    """判断当前问题是否需要改写为独立检索词。"""
    q = query.strip()
    return len(q) <= 10 or any(s in q for s in FOLLOWUP_SIGNALS if len(s) >= 2 and s in q)


def rewrite_query(
    query: str,
    history: list[dict],
    api_key: str,
    base_url: str,
    model_name: str,
) -> str:
    """调用 LLM 将追问改写为独立检索关键词。"""
    # 取最近 6 条历史
    recent = history[-6:] if len(history) > 6 else history
    history_text = "\n".join(
        f"{'👤' if m['role'] == 'user' else '🤖'}: {m['content'][:200]}"
        for m in recent
    )

    prompt = (
        "你的任务是把用户的追问改写成一个独立的、适合搜索文章库的关键词短语。\n"
        "规则：\n"
        "1. 结合对话历史理解用户真正想问什么\n"
        "2. 输出只包含检索关键词，不加任何解释，不加标点\n"
        "3. 关键词要具体，包含核心概念和术语\n"
        "4. 如果问题本身已经完整独立，直接原样返回\n\n"
        f"对话历史：\n{history_text}\n\n"
        f"用户追问：{query}\n\n"
        "检索关键词："
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
            stream=False,
        )
        rewritten = resp.choices[0].message.content.strip()
        # 清理引号和多余标点
        rewritten = rewritten.strip('"\'""''。，,、 ')
        return rewritten if rewritten else query
    except Exception:
        return query  # 降级：用原问题


# ──────────────────────────────────────────────────────────────
# 本地检索
# ──────────────────────────────────────────────────────────────

def retrieve_context(
    query: str, chunks: list[dict], top_k: int = 3
) -> list[dict]:
    """基于关键词重叠分数，检索最相关的 top_k 个文本块。"""
    query_kw = set(extract_keywords(query, top_n=15))

    if not query_kw:
        # 回退：用字符级匹配
        query_kw = set(re.findall(r"[\u4e00-\u9fff]{2,4}", query))

    scored = []
    for chunk in chunks:
        text = chunk["text"]
        score = 0
        # 关键词命中
        for kw in query_kw:
            score += text.count(kw)
        # 完整查询短语命中（加分）
        if query in text:
            score += len(query) * 2
        if score > 0:
            scored.append((score, chunk))

    # 按分数降序
    scored.sort(key=lambda x: -x[0])

    seen_titles = set()
    results = []
    for score, chunk in scored:
        if chunk["title"] not in seen_titles or len(results) < top_k:
            results.append({**chunk, "score": score})
            seen_titles.add(chunk["title"])
        if len(results) >= top_k:
            break

    return results


# ──────────────────────────────────────────────────────────────
# 构建 Prompt
# ──────────────────────────────────────────────────────────────

MODE_PROMPTS = {
    "默认模式": (
        "你是一位深谙投资哲学、个人成长与生活智慧的思想导师。"
        "请基于以下参考文章内容回答问题。"
        "如果参考内容不足以回答，可以结合你的知识补充，但必须标注哪些来自参考、哪些来自自身知识。"
        "回答风格：务实简练，有观点有深度，杜绝空洞说教。\n\n"
    ),
    "硬核投资": (
        "你是一位严谨理性的指数投资策略师。你的核心领域是：指数估值分析、资产配置模型、"
        "ETF 择时策略、网格交易、仓位管理与风险控制。\n"
        "沟通风格：数据驱动，逻辑严密。用估值数字和概率思维说话，不抒情不鸡汤。"
        "多用「低估区域」「安全边际」「均值回归」「标准差」等专业术语。"
        "回答时优先引用参考文章中的具体点位、策略参数和操作逻辑。"
        "如果参考文章不足以支撑判断，明确告知信息不足而非猜测。\n\n"
    ),
    "温暖生活": (
        "你是一位温和睿智的生活导师，擅长用平实的语言疏导焦虑、抚慰心灵。"
        "你的核心议题是：心态调整、焦虑管理、人生感悟、幸福感知、关系经营。\n"
        "沟通风格：温暖但不矫情，有同理心但不滥情。像一位经历丰富的老友在聊天。"
        "用具体的生活场景和朴素的道理说话，不引经据典，不居高临下说教。"
        "回答时优先引用参考文章中的真实感悟和生活智慧。\n\n"
    ),
}


def build_prompt(query: str, contexts: list[dict], mode: str = "默认模式") -> str:
    """将检索到的上下文拼入 System Prompt。"""
    if not contexts:
        return query

    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        context_parts.append(
            f"【参考 {i}】来源：《{ctx['title']}》\n{ctx['text']}"
        )
    context_block = "\n\n".join(context_parts)

    system = MODE_PROMPTS.get(mode, MODE_PROMPTS["默认模式"])
    return f"{system}参考文章：\n{context_block}\n\n用户问题：{query}"


# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_articles():
    """从 JSON 加载文章数据，分词块。"""
    if not DATA_FILE.exists():
        return None, None, [], {}

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # 分类统计
    cat_counts = Counter(a["category"] for a in articles)

    # 全部切块（带分类标签）
    all_chunks = []
    by_category = {"投资逻辑": [], "生活感悟": [], "个人成长": [], "未分类": []}
    for article in articles:
        cat = article["category"]
        chunks = split_into_chunks(
            article["content"], article["title"], chunk_size=500, overlap=80
        )
        for c in chunks:
            c["category"] = cat
        all_chunks.extend(chunks)
        by_category[cat].extend(chunks)

    return articles, cat_counts, all_chunks, by_category


# ──────────────────────────────────────────────────────────────
# UI 渲染
# ──────────────────────────────────────────────────────────────

def render_sidebar(articles, cat_counts):
    """渲染左侧边栏。"""
    # ── 加载配置（优先级：st.secrets > 本地持久化 > 默认值）──
    merged = merge_config()
    saved_provider = merged["provider"]
    saved_api_key = merged["api_key"]
    saved_base_url = merged["base_url"]
    saved_model = merged["model_name"]

    # 初始化 session state（首次加载时恢复所有字段）
    if "init_loaded" not in st.session_state:
        st.session_state["api_key_input"] = saved_api_key
        st.session_state["base_url_input"] = saved_base_url or API_PROVIDERS.get(
            saved_provider, {}
        ).get("base_url", DEFAULT_BASE_URL)
        st.session_state["provider_select"] = saved_provider
        st.session_state["init_loaded"] = True

    provider_keys = list(API_PROVIDERS.keys())
    saved_index = provider_keys.index(saved_provider) if saved_provider in provider_keys else provider_keys.index(DEFAULT_PROVIDER)

    with st.sidebar:
        st.markdown("---")

        # ── WorkBuddy 配置卡片 ──
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                color: white;
            ">
                <h3 style="margin:0 0 4px 0;font-size:16px;">🚀 WorkBuddy 专属配置</h3>
                <p style="margin:0;font-size:12px;opacity:0.85;">连接你的推理引擎</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 提供商选择 ──
        provider = st.selectbox(
            "🤖 API 提供商",
            options=provider_keys,
            index=saved_index,
            key="provider_select",
            help="选择你的 API 服务商，配置自动持久化保存",
        )
        provider_info = API_PROVIDERS[provider]

        # 自定义时才显示完整输入，预设模式自动填充
        if provider == "🔧 自定义":
            api_key = st.text_input(
                "🔑 API Key",
                type="password",
                placeholder="sk-...",
                key="api_key_input",
            )
            base_url = st.text_input(
                "🌐 Base URL",
                value=saved_base_url if not provider_info["base_url"] else provider_info["base_url"],
                placeholder="https://api.xxx.com/v1",
                key="base_url_input",
            )
            model_name = saved_model
        else:
            api_key = st.text_input(
                "🔑 API Key",
                type="password",
                placeholder="sk-...",
                key="api_key_input",
            )
            base_url = st.text_input(
                "🌐 Base URL（自动填充）",
                value=provider_info["base_url"],
                key="base_url_input",
            )
            model_name = provider_info["model"]

        # ── 自动持久化：检测变化即保存 ──
        current_config = {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
        }
        if current_config != merged:
            save_config(current_config)

        # 连接测试
        if st.button("🔍 测试连接", use_container_width=True):
            if not api_key:
                st.warning("请先填入 API Key")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, base_url=base_url)
                    client.models.list()
                    st.success("✅ 连接成功！API 可达")
                except Exception as e:
                    st.error(f"❌ 连接失败: {str(e)[:200]}")

        st.markdown("---")

        # ── 知识台账资产区 ──
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 12px;
                padding: 14px;
                margin-bottom: 12px;
            ">
                <h3 style="margin:0;font-size:15px;color:#333;">📊 知识台账资产区</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if articles is None:
            st.error("⚠️ 未找到 processed_articles.json")
        else:
            total = len(articles)
            st.metric("📚 总文章数", f"{total} 篇")

            # 分类小卡片
            colors = {
                "投资逻辑": "#4A90D9",
                "生活感悟": "#E8856D",
                "个人成长": "#5CB85C",
                "未分类": "#999",
            }

            for cat, color in colors.items():
                cnt = cat_counts.get(cat, 0)
                pct = f"{cnt / total * 100:.1f}%" if total > 0 else "0%"
                st.markdown(
                    f"""
                    <div style="
                        background: white;
                        border-left: 4px solid {color};
                        border-radius: 6px;
                        padding: 10px 12px;
                        margin-bottom: 8px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <span style="font-size:13px;color:#555;">🏷️ {cat}</span>
                        <span style="font-weight:bold;font-size:14px;">{cnt} 篇 <span style="color:#999;font-size:11px;">({pct})</span></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        return api_key, base_url, model_name, provider


def render_main(api_key, base_url, model_name, all_chunks, articles, provider, by_category):
    """渲染主面板。"""
    # 标题
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="font-size:28px; font-weight:700; color:#1a1a2e; margin-bottom:4px;">
                ✨ 专属 AI 思想导师
            </h1>
            <p style="font-size:14px; color:#666;">
                基于 WorkBuddy 读懂博主的投资与生活哲学
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 数据缺失红框警告
    if articles is None:
        st.error(
            """
            ### ❌ 数据底座缺失

            未找到 `processed_articles.json` 文件。

            请先运行数据清洗脚本：
            ```bash
            python clean_data.py
            ```
            确保 `raw_articles/` 目录下有原始文章文件。
            """
        )
        return

    st.markdown("---")

    # ── 阶段切换 ──
    mode = st.radio(
        "🎯 切换对话阶段",
        options=["默认模式", "硬核投资", "温暖生活"],
        horizontal=True,
        index=0,
        key="mode_radio",
        help="切换后 AI 的语气、关注点和专业领域会随之改变",
    )

    # 阶段提示卡片
    if mode == "硬核投资":
        st.info("📈 **硬核投资模式** — 专注指数估值、资产配置与交易策略，回答理性克制，用数据说话。")
    elif mode == "温暖生活":
        st.info("🌿 **温暖生活模式** — 侧重心态调整、焦虑疏导与人生感悟，像一位温和的老友。")
    else:
        st.info("🧭 **默认模式** — 覆盖投资、成长与生活，均衡全面。")

    st.markdown("---")

    # ── 对话历史 ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果是 AI 回复且有引用来源，显示折叠区
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("💡 查看本次回答参考的博主原文出处"):
                    for i, src in enumerate(msg["sources"], 1):
                        cat = src.get("category", "未分类")
                        cat_emoji = {"投资逻辑": "📈", "生活感悟": "🌿", "个人成长": "🧠"}.get(cat, "📄")
                        preview = src["text"][:400]
                        if len(src["text"]) > 400:
                            preview += "…"
                        st.markdown(
                            f"> **📌 原文 {i}**　｜　{cat_emoji} `{cat}`　｜　匹配度 {src.get('score', 0)}\n"
                            f"> \n"
                            f"> **《{src['title']}》**\n"
                            f"> \n"
                            f"> {preview}"
                        )

    # ── 输入框 ──
    placeholder_map = {
        "默认模式": "输入你的问题，探索博主的投资与生活智慧…",
        "硬核投资": "输入投资问题，如：当前中证500估值处于什么位置？如何设计网格策略？…",
        "温暖生活": "写下你的困惑，如：熊市心态崩了怎么办？如何平衡工作与生活？…",
    }
    if prompt := st.chat_input(placeholder_map.get(mode, placeholder_map["默认模式"])):
        # 校验 API Key
        if not api_key:
            st.error("请在左侧边栏填写 WorkBuddy API Key")
            st.stop()

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 检索上下文
        # 按阶段选择检索范围
        if mode == "硬核投资":
            search_chunks = by_category.get("投资逻辑", []) + by_category.get("未分类", [])
        elif mode == "温暖生活":
            search_chunks = (
                by_category.get("生活感悟", [])
                + by_category.get("个人成长", [])
                + by_category.get("未分类", [])
            )
        else:
            search_chunks = all_chunks

        # ── 查询改写：追问时先提炼独立检索词 ──
        search_query = prompt
        if needs_rewrite(prompt) and api_key and base_url and model_name:
            with st.spinner("🔍 正在理解你的追问…"):
                rewritten = rewrite_query(
                    prompt,
                    st.session_state.messages,
                    api_key,
                    base_url,
                    model_name,
                )
                if rewritten != prompt:
                    search_query = rewritten
                    st.caption(f"🔎 检索词已改写: _{rewritten}_")

        contexts = retrieve_context(search_query, search_chunks, top_k=3)
        full_prompt = build_prompt(prompt, contexts, mode=mode)

        # 调用 WorkBuddy API（流式）
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )

                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": MODE_PROMPTS.get(mode, MODE_PROMPTS["默认模式"])},
                        {"role": "user", "content": full_prompt},
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=2048,
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            except Exception as e:
                error_msg = (
                    f"❌ API 调用失败\n\n"
                    f"**请求地址**: `{base_url}`\n\n"
                    f"**错误详情**: {str(e)}"
                )
                message_placeholder.error(error_msg)
                full_response = error_msg

            # 保存 AI 回复及来源
            source_info = [
                {
                    "title": ctx["title"],
                    "category": ctx.get("category", "未分类"),
                    "text": ctx["text"],
                    "score": ctx.get("score", 0),
                }
                for ctx in contexts
            ]
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_info,
            })

            # 显示来源折叠区（当前回答气泡内）
            if contexts:
                with st.expander("💡 查看本次回答参考的博主原文出处"):
                    for i, ctx in enumerate(contexts, 1):
                        cat = ctx.get("category", "未分类")
                        cat_emoji = {"投资逻辑": "📈", "生活感悟": "🌿", "个人成长": "🧠"}.get(cat, "📄")
                        preview = ctx["text"][:400]
                        if len(ctx["text"]) > 400:
                            preview += "…"
                        st.markdown(
                            f"> **📌 原文 {i}**　｜　{cat_emoji} `{cat}`　｜　匹配度 {ctx.get('score', 0)}\n"
                            f"> \n"
                            f"> **《{ctx['title']}》**\n"
                            f"> \n"
                            f"> {preview}"
                        )

    # ── 底部清空按钮 ──
    st.markdown("---")
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        if st.button("🗑️  清空对话", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main():
    # 加载数据
    articles, cat_counts, all_chunks, by_category = load_articles()

    # 左侧栏
    api_key, base_url, model_name, provider = render_sidebar(articles, cat_counts)

    # 主面板
    render_main(api_key, base_url, model_name, all_chunks, articles, provider, by_category)


if __name__ == "__main__":
    main()
