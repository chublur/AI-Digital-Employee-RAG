"""DocumentLoader PDF 分级解析逻辑测试（全 Mock，无真实 PDF/MinerU）。"""
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.document_loader import DocumentLoader


@pytest.fixture
def loader(monkeypatch):
    """使用固定 Settings，避免读取 .env。"""
    test_settings = Settings(
        _env_file=None,
        API_KEY="test",
        MINERU_ENABLED=True,
        MINERU_MIN_CHARS=100,
    )
    monkeypatch.setattr("src.document_loader.settings", test_settings)
    monkeypatch.setattr("src.mineru_loader.settings", test_settings)
    return DocumentLoader()


class TestLoadPdfTiered:
    def test_pymupdf4llm_success_skips_mineru(self, loader):
        long_text = "x" * 200
        with patch("src.document_loader.pymupdf4llm.to_markdown", return_value=long_text):
            text, parser = loader._load_pdf("fake.pdf")
        assert text == long_text
        assert parser == "pymupdf4llm"

    def test_short_pymupdf_triggers_mineru(self, loader):
        with patch("src.document_loader.pymupdf4llm.to_markdown", return_value="短"):
            with patch.object(DocumentLoader, "_load_pdf_with_fitz", return_value=""):
                with patch("src.mineru_loader.is_mineru_available", return_value=True):
                    with patch(
                        "src.mineru_loader.load_pdf_with_mineru",
                        return_value="# MinerU 输出\n\n" + "正文" * 50,
                    ):
                        text, parser = loader._load_pdf("scan.pdf")
        assert parser == "mineru"
        assert "MinerU" in text

    def test_fitz_enough_skips_mineru(self, loader):
        with patch("src.document_loader.pymupdf4llm.to_markdown", return_value="短"):
            with patch.object(
                DocumentLoader,
                "_load_pdf_with_fitz",
                return_value="fitz 提取的正文内容足够长" * 10,
            ):
                text, parser = loader._load_pdf("doc.pdf")
        assert parser == "fitz"
        assert "fitz" in text

    def test_mineru_unavailable_falls_back_to_fitz(self, loader):
        with patch("src.document_loader.pymupdf4llm.to_markdown", return_value=""):
            with patch("src.mineru_loader.is_mineru_available", return_value=False):
                with patch.object(
                    DocumentLoader,
                    "_load_pdf_with_fitz",
                    return_value="fitz 提取的正文内容足够长" * 5,
                ):
                    text, parser = loader._load_pdf("doc.pdf")
        assert parser == "fitz"
        assert "fitz" in text
