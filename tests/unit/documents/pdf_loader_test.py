import sys
import asyncio

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.documents.loader import PDFLoader

# 这一行是为了保证推测出的编码是utf-8


def test_pdf_loader():
    """
    Test: PDFLoader
    """

    pdf_loader = PDFLoader("tests/unit/documents/data/test.pdf")
    pdf_document = asyncio.run(
        pdf_loader.load_file("tests/unit/documents/data/test.pdf")
    )

    assert pdf_document.page_content == "This is a test file."
    assert pdf_document.metadata["src"] == "tests/unit/documents/data/test.pdf"
