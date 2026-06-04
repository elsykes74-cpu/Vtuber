"""Regression tests for the regex sentence segmenter.

These cover a bug where ``segment_text_by_regex`` built its end-punctuation
matcher as a character class (``[...]``) joined with ``|`` instead of a real
alternation. Inside a character class ``|`` is a literal pipe and multi-character
punctuation such as ``"..."`` collapses to a single ``.``. As a result:

* a literal ``|`` in normal text was treated as sentence-ending punctuation and
  split a sentence mid-phrase, and
* an ellipsis ``"..."`` was shattered into three separate ``"."`` fragments.

Each fragment is sent to TTS and shown as its own subtitle, so this corrupted
both spoken audio and on-screen text. The segmenter is reached when
``segment_method`` is ``"regex"`` and also as the automatic fallback from the
pysbd path for unsupported languages or on any pysbd error.
"""

from open_llm_vtuber.utils.sentence_divider import segment_text_by_regex


def test_pipe_is_not_treated_as_sentence_end():
    """A literal '|' must not split a sentence."""
    text = "You can use grep | wc to count lines and then move on."
    sentences, remaining = segment_text_by_regex(text)
    assert sentences == [text]
    assert remaining == ""


def test_multiple_pipes_do_not_split():
    text = "The options are red | green | blue and you should pick one."
    sentences, remaining = segment_text_by_regex(text)
    assert sentences == [text]
    assert remaining == ""


def test_ellipsis_is_not_shattered():
    """'...' must stay attached to its sentence, not become three '.' fragments."""
    sentences, remaining = segment_text_by_regex("Wait... what happened?")
    assert sentences == ["Wait...", "what happened?"]
    assert remaining == ""
    # No empty / punctuation-only fragments should leak through.
    assert all(s.strip(".") for s in sentences)


def test_normal_sentences_still_segment():
    """Ordinary punctuation behaviour is unchanged."""
    sentences, remaining = segment_text_by_regex("Hello world. How are you?")
    assert sentences == ["Hello world.", "How are you?"]
    assert remaining == ""


def test_cjk_full_stop_still_segments():
    sentences, remaining = segment_text_by_regex("这是第一句。这是第二句。")
    assert sentences == ["这是第一句。", "这是第二句。"]
    assert remaining == ""
