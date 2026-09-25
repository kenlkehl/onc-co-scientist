"""Remove only insignificant JSON whitespace from embedded data documents."""
import json


def compact_document(text):
    output = []
    quoted = escaped = False
    for char in text:
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            output.append(char)
        elif char not in ' \t\r\n':
            output.append(char)
    return ''.join(output)


def compact_text(text):
    decoder = json.JSONDecoder()
    output, cursor, scan, documents, removed = [], 0, 0, 0, 0
    while scan < len(text):
        if text[scan] not in '{[':
            scan += 1
            continue
        try:
            value, end = decoder.raw_decode(text, scan)
        except ValueError:
            scan += 1
            continue
        original = text[scan:end]
        compact = compact_document(original)
        # Lexical removal keeps key order, duplicate keys, numeric spelling and
        # every string byte intact; parity also guards the implementation.
        if decoder.decode(compact) != value:
            raise ValueError('JSON whitespace compaction changed a value')
        output.extend([text[cursor:scan], compact])
        removed += len(original) - len(compact)
        documents += 1
        cursor = scan = end
    output.append(text[cursor:])
    return ''.join(output), dict(documents=documents, removed_characters=removed)


def compact_messages(messages):
    result, records = [], []
    for message in messages:
        if not isinstance(message.get('content'), str):
            raise ValueError('Expected frozen text-only vLLM messages')
        content, record = compact_text(message['content'])
        result.append(dict(message, content=content))
        records.append(record)
    return result, records
