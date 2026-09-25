from scripts.gcp_gemma_federation.json_whitespace import compact_text, compact_messages


def test_preserves_narratives_escapes_numeric_spelling_duplicate_keys_and_prose():
    original = 'Intro [not JSON]\n{ "a": 1.00, "a": 2e-3, "text": "x  y\\n\\\" { z }", "b": [ true, null ] }\nEnd'
    compact, audit = compact_text(original)
    assert compact == 'Intro [not JSON]\n{"a":1.00,"a":2e-3,"text":"x  y\\n\\\" { z }","b":[true,null]}\nEnd'
    assert audit['documents'] == 1
    assert audit['removed_characters'] == len(original) - len(compact)
    assert compact_text(compact)[0] == compact


def test_preserves_all_message_fields_and_separate_catalog_documents():
    messages = [dict(role='user', content='Notice\n{ "H": "H123" }\n{ "R": [ "R2", "R3" ] }\nGoal', name='site_1')]
    result, audit = compact_messages(messages)
    assert result == [dict(role='user', content='Notice\n{"H":"H123"}\n{"R":["R2","R3"]}\nGoal', name='site_1')]
    assert audit[0]['documents'] == 2
    assert messages[0]['content'].startswith('Notice\n{ "H"')
