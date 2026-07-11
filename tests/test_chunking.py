from app.ingestion.chunking import chunk_text


def test_chunk_text_basic():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_overlap():
    text = " ".join(str(i) for i in range(1000))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    # consecutive chunks should share overlap words
    first_words = chunks[0].split()[-50:]
    second_words = chunks[1].split()[:50]
    assert first_words == second_words
