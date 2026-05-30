#!/usr/bin/env python3
"""
文章清洗与结构化整理引擎
功能：遍历 raw_articles 文件夹，自动识别 .txt/.md/.pdf/.html，提取纯文本，
      智能分类，输出 JSON（RAG 底座）+ CSV（台账）双重结果。
"""

import os
import re
import json
import csv
import sys
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────
# 依赖检测 & 导入
# ──────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_articles"
OUTPUT_JSON = BASE_DIR / "processed_articles.json"
OUTPUT_CSV = BASE_DIR / "文章资产总台账.csv"

SUPPORTED_EXT = {".txt", ".md", ".pdf", ".html"}
SKIP_NAMES = {".DS_Store", "Thumbs.db"}

# ──────────────────────────────────────────────────────────────
# 分类关键词（可自行扩展）
# ──────────────────────────────────────────────────────────────
CATEGORY_RULES = [
    ("投资逻辑", [
        "基金", "ETF", "投资", "估值", "网格", "资产", "指数", "交易",
        "策略", "收益", "风险", "配置", "仓位", "波段", "牛市", "熊市",
        "定投", "抄底", "止盈", "分红", "持仓", "买入", "卖出", "调仓",
        "低估", "高估", "复利", "套利", "对冲", "杠杆", "美股", "港股",
        "A股", "纳斯达克", "恒生", "标普", "均线", "趋势", "支撑",
        "压力", "MACD", "K线", "净值", "年化", "回撤", "波动",
    ]),
    ("个人成长", [
        "认知", "习惯", "自律", "读书", "时间管理", "成长", "学习",
        "思维", "体系", "方法", "专注", "效率", "反思", "规划",
        "提升", "目标", "执行", "坚持", "努力", "积累",
    ]),
    ("生活感悟", [
        "焦虑", "心态", "生活", "旅行", "感悟", "幸福", "婚姻",
        "家庭", "健康", "情感", "孤独", "自由", "快乐", "痛苦",
        "人生", "朋友", "父母", "孩子", "过去", "未来", "命运",
        "善良", "温暖", "平静", "珍惜",
    ]),
]

# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def safe_read_text(filepath: Path) -> str:
    """多编码容错读取文本文件。"""
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            return filepath.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 兜底：ignore errors
    return filepath.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(filepath: Path) -> str:
    """使用 pypdf 提取 PDF 纯文本。"""
    if not HAS_PYPDF:
        raise RuntimeError("pypdf 未安装，无法处理 PDF 文件。")
    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_html(filepath: Path) -> str:
    """使用 BeautifulSoup 剥离 HTML 标签，只保留纯文本。"""
    raw = safe_read_text(filepath)
    if HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")
        # 移除 script / style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        # 无 bs4 时简单正则去标签
        text = re.sub(r"<[^>]+>", "", raw)
    # 清理多余空行
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def read_content(filepath: Path) -> tuple[str, str]:
    """
    读取文件内容，返回 (纯文本, 格式类型)。
    格式类型: txt / md / pdf / html
    """
    ext = filepath.suffix.lower()

    if ext in (".txt", ".md"):
        return safe_read_text(filepath), ext[1:]  # 去掉点号

    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
        return text, "pdf"

    if ext == ".html":
        text = extract_text_from_html(filepath)
        return text, "html"

    raise ValueError(f"不支持的文件类型: {ext}")


def extract_title(filename: str) -> str:
    """从文件名提取标题，去除后缀（包括 .pdf.pdf 双后缀）。"""
    name = filename
    # 递归去除已知后缀
    while True:
        changed = False
        for ext in [".txt", ".md", ".pdf", ".html"]:
            if name.lower().endswith(ext):
                name = name[:-len(ext)]
                changed = True
        if not changed:
            break
    return name.strip()


def count_words(text: str) -> int:
    """计算文本字数（中文字符 + 英文单词）。"""
    # 中文字符数
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    # 英文单词数
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return chinese_chars + english_words


def classify_article(text: str) -> str:
    """
    基于关键词匹配自动归类。
    多标签命中时，按命中数最多 → 靠前规则 返回。
    """
    scores = {}
    for label, keywords in CATEGORY_RULES:
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[label] = score

    if not scores:
        return "未分类"

    # 取最高分标签
    best = max(scores, key=lambda k: scores[k])
    return best


# ──────────────────────────────────────────────────────────────
# 进度条
# ──────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, bar_len: int = 40) -> str:
    """生成进度条字符串。"""
    pct = current / total if total > 0 else 1.0
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  📦 文章清洗与结构化整理引擎  v1.0")
    print("=" * 60)
    print()

    # ── 1. 收集文件 ──────────────────────────────────────────
    all_files = []
    for f in RAW_DIR.rglob("*"):
        if f.is_file() and f.name not in SKIP_NAMES:
            ext = f.suffix.lower()
            if ext in SUPPORTED_EXT or f.name.lower().endswith((".pdf.pdf.txt", ".pdf.txt")):
                all_files.append(f)

    total = len(all_files)
    print(f"  📂 扫描完成，发现 {total} 个待处理文件")
    print(f"     输出 JSON: {OUTPUT_JSON}")
    print(f"     输出 CSV:  {OUTPUT_CSV}")
    print()

    if total == 0:
        print("  ⚠️  没有找到可处理的文件，退出。")
        return

    # ── 2. 逐文件处理 ────────────────────────────────────────
    articles = []       # 完整数据 (用于 JSON)
    csv_rows = []       # 台账数据 (用于 CSV)
    stats = {"txt": 0, "md": 0, "pdf": 0, "html": 0, "fail": 0}

    for idx, filepath in enumerate(all_files, 1):
        filename = filepath.name
        bar = progress_bar(idx, total)

        try:
            # 读取内容
            text, fmt = read_content(filepath)

            if not text or not text.strip():
                print(f"\r  {bar}  跳过 {filename} — 内容为空", end="")
                stats["fail"] += 1
                continue

            # 提取特征
            title = extract_title(filename)
            word_count = count_words(text)
            category = classify_article(text)

            # 修正格式类型（.pdf.pdf.txt → pdf）
            if "pdf" in filename.lower() and fmt in ("txt", "md"):
                fmt = "pdf"

            # 打印日志
            print(f"\r  {bar}  ✅ {filename} → [{category}] {word_count}字")

            # 记录
            articles.append({
                "title": title,
                "filename": filename,
                "format": fmt,
                "category": category,
                "word_count": word_count,
                "content": text.strip(),
            })

            csv_rows.append({
                "title": title,
                "format": fmt,
                "word_count": word_count,
                "category": category,
            })

            stats[fmt] = stats.get(fmt, 0) + 1

        except Exception as e:
            print(f"\r  {bar}  ❌ {filename} 失败: {e}")
            stats["fail"] += 1

    print()
    print()

    # ── 3. 输出 JSON ─────────────────────────────────────────
    print("  💾 正在写入 processed_articles.json ...", end=" ")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"完成 ({len(articles)} 篇)")

    # ── 4. 输出 CSV ─────────────────────────────────────────
    print("  💾 正在写入 文章资产总台账.csv ...", end=" ")
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "format", "word_count", "category"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"完成 ({len(csv_rows)} 行)")

    print()
    print("=" * 60)

    # ── 5. 战报 ──────────────────────────────────────────────
    total_success = total - stats["fail"]
    print(f"  🎉 处理完成！")
    success_parts = []
    for fmt_name in ["txt", "md", "pdf", "html"]:
        cnt = stats.get(fmt_name, 0)
        if cnt > 0:
            success_parts.append(f"{cnt} 篇 {fmt_name.upper()}")
    print(f"     共成功清洗 {'，'.join(success_parts)}")
    if stats["fail"] > 0:
        print(f"     ⚠️  失败 {stats['fail']} 篇")
    print()

    # 分类统计
    cat_counts = {}
    for row in csv_rows:
        cat_counts[row["category"]] = cat_counts.get(row["category"], 0) + 1
    print("  📊 分类统计:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"     [{cat}]  {cnt} 篇")

    print()
    print("  📋 总台账已生成，可用 Excel 打开查看。")
    print("=" * 60)


if __name__ == "__main__":
    main()
