"""
MinerU PDF 解析兜底（可选依赖）。

当 pymupdf4llm 提取内容过少（常见于扫描件、复杂版式）时，
调用本地 MinerU CLI 做 OCR + 版式重建。

MinerU 3.x 默认输出 JSON（content_list / middle），不一定有 .md；
本模块会依次尝试 .md → content_list JSON → middle JSON。

安装：pip install "mineru[core]"

本地提速建议：
1. 常驻 mineru-api，设置 MINERU_API_URL，避免每次冷启动（可省 20~30s）
2. MINERU_FORMULA=false、MINERU_TABLE=false（不需要公式/表格时）
3. MINERU_METHOD=txt（纯文字层 PDF）或 auto（自动）
4. MINERU_START_PAGE / MINERU_END_PAGE 限制页数
5. 有 NVIDIA GPU 时可设 MINERU_BACKEND=hybrid-engine
"""
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)


def is_mineru_available() -> bool:
    """检测 mineru CLI 是否在 PATH 中。"""
    return shutil.which("mineru") is not None


def _collect_texts(obj: Any, texts: list[str]) -> None:
    """递归从 MinerU JSON 结构中提取文本字段。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("text", "content") and isinstance(value, str) and value.strip():
                # 跳过页码等噪声
                if key == "text" and obj.get("type") == "page_number":
                    continue
                texts.append(value.strip())
            else:
                _collect_texts(value, texts)
    elif isinstance(obj, list):
        for item in obj:
            _collect_texts(item, texts)


def _extract_from_json_files(output_dir: Path) -> str:
    """从 MinerU 3.x 的 content_list / middle JSON 还原 Markdown 风格正文。"""
    candidates: list[Path] = []
    candidates.extend(sorted(output_dir.rglob("*_content_list_v2.json"), reverse=True))
    candidates.extend(sorted(output_dir.rglob("*_content_list.json"), reverse=True))
    candidates.extend(sorted(output_dir.rglob("*_middle.json"), reverse=True))

    for json_path in candidates:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"读取 MinerU JSON 失败 {json_path}: {e}")
            continue

        texts: list[str] = []
        _collect_texts(data, texts)
        # 去重保序
        seen = set()
        unique = []
        for t in texts:
            if t not in seen and len(t) > 1:
                seen.add(t)
                unique.append(t)

        body = "\n\n".join(unique).strip()
        if len(body) >= 10:
            logger.info(f"MinerU JSON 提取: {json_path.name} ({len(body)} 字)")
            return body
    return ""


def _find_markdown_output(output_dir: Path) -> str:
    """
    在 MinerU 输出目录中查找正文。
    优先 .md，其次 JSON（MinerU 3.x 常见）。
    """
    md_files = sorted(output_dir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) >= 10:
            logger.info(f"MinerU Markdown: {md_path} ({len(text)} 字)")
            return text

    return _extract_from_json_files(output_dir)


def _build_mineru_cmd(pdf_path: Path, output_dir: str, method: Optional[str] = None) -> list[str]:
    """组装 MinerU CLI 参数（含提速选项）。"""
    parse_method = method or settings.MINERU_METHOD
    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "-o", output_dir,
        "-b", settings.MINERU_BACKEND,
        "-m", parse_method,
        "-f", str(settings.MINERU_FORMULA).lower(),
        "-t", str(settings.MINERU_TABLE).lower(),
        "--client-side-output-generation",
        str(settings.MINERU_CLIENT_SIDE_MD).lower(),
    ]

    if settings.MINERU_LANGUAGE:
        cmd.extend(["-l", settings.MINERU_LANGUAGE])

    if settings.MINERU_START_PAGE is not None:
        cmd.extend(["-s", str(settings.MINERU_START_PAGE)])
    if settings.MINERU_END_PAGE is not None:
        cmd.extend(["-e", str(settings.MINERU_END_PAGE)])

    if settings.MINERU_API_URL:
        cmd.extend(["--api-url", settings.MINERU_API_URL])

    return cmd


def load_pdf_with_mineru(file_path: str, method: Optional[str] = None) -> str:
    """
    调用 MinerU CLI 解析 PDF，返回 Markdown/文本字符串。

    Args:
        file_path: PDF 路径
        method: 覆盖 MINERU_METHOD（ocr/txt/auto），扫描件建议 ocr

    Raises:
        RuntimeError: CLI 执行失败或未找到有效输出
        FileNotFoundError: mineru 命令不存在
    """
    if not is_mineru_available():
        raise FileNotFoundError(
            "未找到 mineru 命令。请安装: pip install \"mineru[core]\""
        )

    pdf_path = Path(file_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    with tempfile.TemporaryDirectory(prefix="mineru_out_") as tmp_out:
        cmd = _build_mineru_cmd(pdf_path, tmp_out, method=method)
        logger.info(
            f"MinerU 解析: backend={settings.MINERU_BACKEND}, "
            f"method={method or settings.MINERU_METHOD}, file={pdf_path.name}"
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=settings.MINERU_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"MinerU 解析超时（>{settings.MINERU_TIMEOUT}s），"
                "可增大 MINERU_TIMEOUT 或设置 MINERU_START_PAGE/MINERU_END_PAGE 限制页数"
            ) from e

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "未知错误").strip()
            raise RuntimeError(f"MinerU CLI 失败 (code={result.returncode}): {err[:500]}")

        text = _find_markdown_output(Path(tmp_out))
        if not text:
            raise RuntimeError("MinerU 未产生有效输出（无 .md 且 JSON 为空）")
        return text
