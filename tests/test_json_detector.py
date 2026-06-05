from open_llm_vtuber.mcpp.json_detector import StreamJSONDetector


def test_process_chunk_ignores_braces_inside_strings():
    detector = StreamJSONDetector()

    result = detector.process_chunk('prefix {"text": "literal } brace", "ok": true} suffix')

    assert result == [{"text": "literal } brace", "ok": True}]


def test_process_chunk_handles_escaped_quotes_before_braces():
    detector = StreamJSONDetector()

    result = detector.process_chunk('{"text": "quoted \\"}\\"", "ok": true}')

    assert result == [{"text": 'quoted "}"', "ok": True}]
