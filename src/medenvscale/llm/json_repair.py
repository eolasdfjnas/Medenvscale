from __future__ import annotations

import ast
import json


def parse_json_payload(text: str) -> dict:
    candidate = _extract_outer_json_object(text.strip())
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _escape_invalid_control_chars(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        repaired = _strip_trailing_commas(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pythonish = _replace_json_literals_outside_strings(repaired)
        try:
            parsed = ast.literal_eval(pythonish)
        except (SyntaxError, ValueError) as exc:
            raise json.JSONDecodeError(str(exc), pythonish, 0) from exc
        if isinstance(parsed, dict):
            return parsed
        raise json.JSONDecodeError("Parsed payload is not a JSON object", pythonish, 0)


def _extract_outer_json_object(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text


def _escape_invalid_control_chars(text: str) -> str:
    output: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                output.append(char)
                escape = False
                continue
            if char == "\\":
                output.append(char)
                escape = True
                continue
            if char == '"':
                output.append(char)
                in_string = False
                continue
            if char == "\n":
                output.append("\\n")
                continue
            if char == "\r":
                output.append("\\r")
                continue
            if char == "\t":
                output.append("\\t")
                continue
            if ord(char) < 0x20:
                output.append(f"\\u{ord(char):04x}")
                continue
            output.append(char)
            continue

        output.append(char)
        if char == '"':
            in_string = True
            escape = False

    return "".join(output)


def _strip_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escape = False
    length = len(text)

    for index, char in enumerate(text):
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            output.append(char)
            continue

        if char == ",":
            next_index = index + 1
            while next_index < length and text[next_index] in " \t\r\n":
                next_index += 1
            if next_index < length and text[next_index] in "]}":
                continue

        output.append(char)

    return "".join(output)


def _replace_json_literals_outside_strings(text: str) -> str:
    output: list[str] = []
    in_string = False
    escape = False
    length = len(text)
    index = 0

    while index < length:
        char = text[index]
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if text.startswith("true", index):
            output.append("True")
            index += 4
            continue
        if text.startswith("false", index):
            output.append("False")
            index += 5
            continue
        if text.startswith("null", index):
            output.append("None")
            index += 4
            continue

        output.append(char)
        index += 1

    return "".join(output)
