"""MinerU 试用脚本：验证安装、解析与耗时。

用法：
    python scripts/test_mineru.py test/sample_text.pdf
    python scripts/test_mineru.py your.pdf --force-mineru
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.document_loader import DocumentLoader
from src.mineru_loader import is_mineru_available, load_pdf_with_mineru


def main():
    parser = argparse.ArgumentParser(description="MinerU 解析试用")
    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument("--force-mineru", action="store_true", help="跳过 pymupdf4llm，直接测 MinerU")
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default=None)
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"文件不存在: {pdf}")
        sys.exit(1)

    print(f"mineru CLI: {'已安装' if is_mineru_available() else '未安装'}")
    print(f"backend={settings.MINERU_BACKEND}, method={args.method or settings.MINERU_METHOD}")
    print(f"formula={settings.MINERU_FORMULA}, table={settings.MINERU_TABLE}")
    if settings.MINERU_API_URL:
        print(f"api_url={settings.MINERU_API_URL} (常驻服务，更快)")

    if args.force_mineru:
        t0 = time.perf_counter()
        try:
            text = load_pdf_with_mineru(str(pdf), method=args.method)
            print(f"\nMinerU 成功: {len(text)} 字, 耗时 {time.perf_counter()-t0:.1f}s")
            print("预览:", text[:300].replace("\n", " "))
        except Exception as e:
            print(f"\nMinerU 失败 ({time.perf_counter()-t0:.1f}s): {e}")
            sys.exit(1)
        return

    loader = DocumentLoader()
    t0 = time.perf_counter()
    try:
        docs = loader.load_and_split(str(pdf))
    except Exception as e:
        print(f"解析失败 ({time.perf_counter()-t0:.1f}s): {e}")
        sys.exit(1)

    parser_name = docs[0].metadata.get("pdf_parser", "?")
    print(f"\n入库成功: {len(docs)} 块, parser={parser_name}, 耗时 {time.perf_counter()-t0:.1f}s")
    print("首块预览:", docs[0].page_content[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
