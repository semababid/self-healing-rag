import pytest

from src import ingest, vectorstore
from src.service import add_document, remove_uploads


def test_add_document_saves_to_the_upload_folder(upload_dir):
    path = add_document("menu.md", b"# Menu\nFish soup on Fridays.")
    assert path == upload_dir / "menu.md"
    assert path.read_bytes() == b"# Menu\nFish soup on Fridays."


def test_add_document_drops_folder_parts_from_the_name(upload_dir):
    path = add_document("../../escape.md", b"x")
    assert path == upload_dir / "escape.md"


@pytest.mark.parametrize("name, message", [
    (".hidden.md", "hidden files"),
    ("", "hidden files"),
    ("notes.exe", "only .md, .txt and .pdf"),
    ("photo.png", "only .md, .txt and .pdf"),
])
def test_add_document_rejects_other_files(upload_dir, name, message):
    with pytest.raises(ValueError, match=message):
        add_document(name, b"x")
    assert not upload_dir.exists() or not any(upload_dir.iterdir())


def test_uploaded_documents_are_ingested_and_can_be_removed(temp_store):
    add_document("menu.md", b"# Canteen menu\nOn Fridays there is fish soup.")
    ingest.ingest()
    texts = vectorstore.get_vectorstore().get()["documents"]
    assert any("fish soup" in text for text in texts)

    assert remove_uploads() == 1
    ingest.ingest()
    texts = vectorstore.get_vectorstore().get()["documents"]
    assert not any("fish soup" in text for text in texts)


def test_remove_uploads_with_nothing_uploaded():
    assert remove_uploads() == 0
