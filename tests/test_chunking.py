def test_chunk_text_creates_expected_spans(prompt_guard):
    text = "a b c d e f g h i j"
    spans = prompt_guard._chunk_text(text, max_seq_length=4, overlap=1)
    assert spans == [(0, 7), (6, 13), (12, 19)]


def test_chunk_text_union_covers_tokens(prompt_guard):
    text = "a b c d e f g h i j"
    spans = prompt_guard._chunk_text(text, max_seq_length=4, overlap=1)
    covered = set()
    for start, end in spans:
        covered.update(range(start, end))
    assert covered == set(range(0, len(text)))


def test_chunk_text_consecutive_spans_overlap(prompt_guard):
    text = "a b c d e f"
    spans = prompt_guard._chunk_text(text, max_seq_length=4, overlap=2)
    assert spans[0][1] > spans[1][0]


def test_chunk_text_single_chunk(prompt_guard):
    text = "hello world"
    spans = prompt_guard._chunk_text(text, max_seq_length=64, overlap=0)
    assert spans == [(0, 11)]


def test_chunk_text_empty_text(prompt_guard):
    assert prompt_guard._chunk_text("", max_seq_length=64, overlap=0) == []


def test_chunk_text_invalid_overlap_raises(prompt_guard):
    try:
        prompt_guard._chunk_text("hello world", max_seq_length=4, overlap=4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_chunk_text_invalid_overlap_greater_raises(prompt_guard):
    try:
        prompt_guard._chunk_text("hello world", max_seq_length=4, overlap=8)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
