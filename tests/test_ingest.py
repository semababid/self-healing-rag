from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src import config, ingest, vectorstore


def make_pdf(path: Path, text: str) -> None:
    """Write a one-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(out)


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "handbook.md").write_text("The N7 has a maximum flight time of 38 minutes.\n" * 40)
    (folder / "parts.txt").write_text("Replacement parts are shipped from Tallinn.")
    (folder / ".DS_Store").write_bytes(b"\x00\x01")
    monkeypatch.setattr(config, "DATA_DIR", folder)
    return folder


@pytest.fixture
def temp_store(tmp_path, monkeypatch, docs_dir):
    # A store in a temp folder with fake embeddings: no model download, real chroma_db untouched.
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))


def test_load_file_reads_markdown(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\nSome text.")
    docs = ingest.load_file(path)
    assert len(docs) == 1
    assert docs[0].page_content == "# Notes\nSome text."
    assert docs[0].metadata == {"source": str(path)}


def test_load_file_reads_a_pdf_page(tmp_path):
    path = tmp_path / "manual.pdf"
    make_pdf(path, "Recalibration takes about 90 seconds")
    docs = ingest.load_file(path)
    assert [doc.page_content.strip() for doc in docs] == ["Recalibration takes about 90 seconds"]
    assert docs[0].metadata == {"source": str(path), "page": 0}


def test_load_file_skips_other_file_types(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(b"\x89PNG")
    assert ingest.load_file(path) == []


def test_load_documents_skips_hidden_files(docs_dir):
    names = sorted(Path(doc.metadata["source"]).name for doc in ingest.load_documents())
    assert names == ["handbook.md", "parts.txt"]


def test_load_documents_fails_on_an_empty_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        ingest.load_documents()


def test_ingesting_twice_doesnt_duplicate_chunks(temp_store):
    ingest.ingest()
    first = len(vectorstore.get_vectorstore().get()["ids"])
    ingest.ingest()
    second = len(vectorstore.get_vectorstore().get()["ids"])
    assert first == second > 1


def test_require_store_built(temp_store):
    with pytest.raises(RuntimeError):
        vectorstore.require_store_built()
    ingest.ingest()
    vectorstore.require_store_built()


def test_ingested_chunks_can_be_retrieved(temp_store):
    ingest.ingest()
    docs = vectorstore.get_retriever(k=2).invoke("Where are replacement parts shipped from?")
    assert len(docs) == 2
