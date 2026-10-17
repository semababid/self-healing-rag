from pathlib import Path

import pytest
from langchain_core.documents import Document

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


def test_load_file_reads_markdown(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\nSome text.")
    docs = ingest.load_file(path)
    assert len(docs) == 1
    assert docs[0].page_content == "# Notes\nSome text."
    assert docs[0].metadata == {"source": str(path), "title": "Notes"}


def test_load_file_reads_a_pdf_page(tmp_path):
    path = tmp_path / "manual.pdf"
    make_pdf(path, "Recalibration takes about 90 seconds")
    docs = ingest.load_file(path)
    assert [doc.page_content.strip() for doc in docs] == ["Recalibration takes about 90 seconds"]
    assert docs[0].metadata == {"source": str(path), "title": "manual", "page": 0}


def test_title_of_uses_the_first_heading_or_the_file_name():
    assert ingest.title_of("intro\n# Nimbus N9: Preview\n## Specs", Path("n9.md")) == "Nimbus N9: Preview"
    assert ingest.title_of("no heading here", Path("notes.txt")) == "notes"


def test_every_chunk_starts_with_its_document_title(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 120)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    text = "# Nimbus N9: Preview\n\n" + "\n".join(f"- Spec line {n} of the N9." for n in range(12))
    doc = Document(page_content=text, metadata={"source": "n9.md", "title": "Nimbus N9: Preview"})

    chunks = ingest.split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_content.split("\n", 1)[0] in ("# Nimbus N9: Preview", "Nimbus N9: Preview")


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


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_ingested_chunks_can_be_searched(temp_store, monkeypatch, mode):
    monkeypatch.setattr(config, "SEARCH", mode)
    ingest.ingest()
    docs = vectorstore.search("Where are replacement parts shipped from?", k=2)
    assert len(docs) == 2


def chunks(*texts):
    return [Document(page_content=text) for text in texts]


def test_fuse_prefers_chunks_ranked_well_in_both_lists():
    by_vector = chunks("a", "b", "c")
    by_keyword = chunks("b", "c", "a")
    assert [doc.page_content for doc in vectorstore.fuse([by_vector, by_keyword], k=3)] == ["b", "a", "c"]


def test_fuse_keeps_chunks_found_by_only_one_list():
    fused = vectorstore.fuse([chunks("a", "b"), chunks("c")], k=3)
    assert sorted(doc.page_content for doc in fused) == ["a", "b", "c"]


def test_fuse_returns_at_most_k():
    assert len(vectorstore.fuse([chunks("a", "b", "c"), chunks("c", "b", "a")], k=2)) == 2


def test_tokenize():
    assert vectorstore.tokenize("Check zone.cfg, then 20%!") == ["check", "zone", "cfg", "then", "20"]
