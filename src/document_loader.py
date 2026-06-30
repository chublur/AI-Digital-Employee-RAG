"""
文档解析模块。

支持格式：
  .pdf   → pymupdf4llm（主）→ MinerU（扫描件/复杂版式兜底）→ fitz（最后兜底）
  .docx  → python-docx 提取正文段落、标题、表格

解析后统一转为 LangChain Document 列表，交给下游分片器处理。
"""
import logging
from pathlib import Path
from typing import List, Tuple

import fitz
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class DocumentLoader:
    def __init__(self):
        self.chunk_size = getattr(settings, "CHUNK_SIZE", 800)
        self.chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 150)

        headers_to_split_on = [("#", "H1"), ("##", "H2")]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()
        logger.info(f"开始解析文档: {file_path_obj.name} (格式: {suffix})")

        pdf_parser = ""
        if suffix == ".pdf":
            text, pdf_parser = self._load_pdf(file_path)
        elif suffix == ".docx":
            text = self._load_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，当前支持: {SUPPORTED_EXTENSIONS}")

        if not text or not text.strip():
            raise ValueError(
                f"文档解析结果为空: {file_path_obj.name}。"
                "若为扫描件 PDF，请安装 MinerU: pip install \"mineru[core]\""
            )

        # 统一分片流程
        header_splits = self.header_splitter.split_text(text)
        final_splits = self.text_splitter.split_documents(header_splits)

        for i, doc in enumerate(final_splits):
            meta = {
                "source": file_path_obj.name,
                "chunk_id": i,
                "file_type": suffix.lstrip("."),
                "has_table": "|" in doc.page_content,
            }
            if pdf_parser:
                meta["pdf_parser"] = pdf_parser
            doc.metadata.update(meta)

        logger.info(
            f"解析完成: {file_path_obj.name} → {len(final_splits)} 个文本块"
            + (f" (parser={pdf_parser})" if pdf_parser else "")
        )
        return final_splits

    # ── PDF 解析 ──────────────────────────────────────────

    def _load_pdf(self, file_path: str) -> Tuple[str, str]:
        """
        PDF → Markdown 字符串。

        分级策略：
        1. pymupdf4llm — 快，适合电子版 PDF
        2. MinerU — 内容过少或主路径失败时触发（扫描件 OCR、复杂版式）
        3. fitz — 最后兜底，纯文本提取

        Returns:
            (markdown_text, parser_name)
        """
        min_chars = settings.MINERU_MIN_CHARS

        # ── 1. pymupdf4llm ──
        pymupdf_text = ""
        try:
            pymupdf_text = pymupdf4llm.to_markdown(str(file_path)) or ""
            if len(pymupdf_text.strip()) >= min_chars:
                return pymupdf_text, "pymupdf4llm"
            logger.warning(
                f"pymupdf4llm 提取内容过少 ({len(pymupdf_text.strip())} 字 < {min_chars})，"
                "尝试 MinerU 兜底"
            )
        except Exception as e:
            logger.warning(f"pymupdf4llm 解析失败: {e}")

        # ── 2. MinerU（扫描件 / 复杂 PDF，fitz 也提取不足时才触发）──
        fitz_preview = ""
        try:
            fitz_preview = self._load_pdf_with_fitz(file_path).strip()
        except Exception as e:
            logger.debug(f"fitz 预检失败: {e}")

        if len(fitz_preview) >= min_chars:
            logger.info(
                f"pymupdf4llm 内容不足但 fitz 可提取 ({len(fitz_preview)} 字)，"
                "跳过 MinerU 以节省时间"
            )
            return fitz_preview, "fitz"

        if settings.MINERU_ENABLED:
            try:
                from src.mineru_loader import is_mineru_available, load_pdf_with_mineru

                if is_mineru_available():
                    # 几乎无文字层 → 强制 OCR；否则用配置的 auto/txt
                    mineru_method = (
                        "ocr"
                        if len(pymupdf_text.strip()) < 20
                        else None
                    )
                    mineru_text = load_pdf_with_mineru(file_path, method=mineru_method)
                    if len(mineru_text.strip()) >= 10:
                        return mineru_text, "mineru"
                    logger.warning("MinerU 输出内容过少")
                else:
                    logger.warning(
                        "MINERU_ENABLED=true 但未安装 mineru CLI，"
                        "跳过 MinerU 兜底（pip install \"mineru[core]\"）"
                    )
            except Exception as e:
                logger.warning(f"MinerU 解析失败: {e}")

        # ── 3. fitz 纯文本兜底 ──
        try:
            fitz_text = self._load_pdf_with_fitz(file_path)
            if len(fitz_text.strip()) >= 10:
                return fitz_text, "fitz"
        except Exception as e:
            logger.warning(f"fitz 文本提取失败: {e}")

        # pymupdf 有少量内容时也返回，避免完全失败
        if pymupdf_text.strip():
            return pymupdf_text, "pymupdf4llm"

        raise ValueError(
            "PDF 解析失败：未能提取有效文本。"
            "若为扫描件，请安装 MinerU 并确保 MINERU_ENABLED=true。"
        )

    @staticmethod
    def _load_pdf_with_fitz(file_path: str) -> str:
        """fitz 逐页纯文本提取（无 OCR，仅读文字层）。"""
        doc = fitz.open(str(file_path))
        try:
            return "\n\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    # ── Word 解析 ─────────────────────────────────────────

    def _load_docx(self, file_path: str) -> str:
        """
        Word .docx → Markdown 字符串。

        提取规则：
        - Heading 1/2/3 样式 → ## / ### / #### 标题
        - 普通段落 → 原文保留
        - 表格 → Markdown 表格语法（首行为表头）
        - 忽略页眉、页脚、图片（只取文字）

        为什么转成 Markdown？
          让 MarkdownHeaderTextSplitter 能按标题层级分片，
          和 PDF 走相同的下游处理流程，不需要额外分支。
        """
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError(
                "python-docx 未安装，请运行: pip install python-docx>=1.1.0"
            )

        docx = DocxDocument(file_path)
        lines: List[str] = []

        # 按文档顺序遍历段落和表格（docx 的 body 元素顺序即原始顺序）
        for element in docx.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # 段落
                para_text = element.text_content() if hasattr(element, "text_content") else ""
                # 用 lxml 的 itertext 获取完整文本（含 run 中的内容）
                para_text = "".join(element.itertext()).strip()
                if not para_text:
                    continue

                # 识别标题样式
                style_name = ""
                pPr = element.find(f".//{{{element.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}pStyle")
                if pPr is not None:
                    style_name = pPr.get(f"{{{element.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}val", "")

                if style_name.startswith("Heading1") or style_name == "1":
                    lines.append(f"# {para_text}")
                elif style_name.startswith("Heading2") or style_name == "2":
                    lines.append(f"## {para_text}")
                elif style_name.startswith("Heading3") or style_name == "3":
                    lines.append(f"### {para_text}")
                else:
                    lines.append(para_text)

            elif tag == "tbl":
                # 表格 → Markdown 表格
                table_lines = self._docx_table_to_markdown(element)
                lines.extend(table_lines)
                lines.append("")  # 表格后空行

        return "\n\n".join(lines)

    def _docx_table_to_markdown(self, tbl_element) -> List[str]:
        """
        将 docx XML 表格元素转为 Markdown 表格行列表。
        首行视为表头，插入分隔线。
        """
        try:
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            rows = tbl_element.findall(f".//{{{ns}}}tr")
            if not rows:
                return []

            md_rows: List[str] = []
            for i, row in enumerate(rows):
                cells = row.findall(f".//{{{ns}}}tc")
                cell_texts = ["".join(c.itertext()).strip() for c in cells]
                md_rows.append("| " + " | ".join(cell_texts) + " |")
                if i == 0:
                    # 表头分隔线
                    md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
            return md_rows
        except Exception as e:
            logger.warning(f"表格转换失败，跳过: {e}")
            return []
