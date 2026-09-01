# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #92 — tool arguments must land on their declared type.

The XML-ish tool dialects (qwen3_coder, the parser on the production coding
route) carry every ``<parameter=name>`` value as raw TEXT, so an array/object
parameter arrives as a *string* even when the model wrote valid JSON.
``_coerce_tool_arguments`` previously handled only the opposite direction, so
the client got a string where its schema declares an array and rejected an
otherwise-correct call.
"""

import json

import pytest

import vllm_mlx.server as srv


def _tool(name="todowrite", todos_type="array"):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {"type": todos_type},
                        "note": {"type": "string"},
                    },
                },
            },
        }
    ]


def coerce(args: dict, tools=None, name="todowrite"):
    out = srv._coerce_tool_arguments(json.dumps(args), name, tools or _tool())
    return json.loads(out)


# ----------------------------------------------------- the #92 direction

TODOS = [
    {"content": "Create feature branch off main", "status": "in_progress"},
    {"content": "Open PR to main", "status": "pending"},
]


def test_json_string_becomes_the_declared_array():
    """The live opencode todowrite shape."""
    got = coerce({"todos": json.dumps(TODOS)})
    assert isinstance(got["todos"], list)
    assert got["todos"] == TODOS


def test_leading_and_trailing_whitespace_is_tolerated():
    """The parser hands the value over with the surrounding newlines intact."""
    got = coerce({"todos": "\n" + json.dumps(TODOS) + "\n"})
    assert got["todos"] == TODOS


def test_json_string_becomes_the_declared_object():
    payload = {"a": 1, "b": [2, 3]}
    got = coerce({"todos": json.dumps(payload)}, tools=_tool(todos_type="object"))
    assert got["todos"] == payload


def test_union_typed_schema_still_coerces():
    tools = _tool()
    tools[0]["function"]["parameters"]["properties"]["todos"]["type"] = [
        "array",
        "null",
    ]
    got = coerce({"todos": json.dumps(TODOS)}, tools=tools)
    assert got["todos"] == TODOS


# ------------------------------------------- what must NOT be touched


def test_malformed_json_is_left_alone_not_repaired():
    """No fabrication: a dangling key is a decoding problem, not a parsing one.

    This is the exact live failure — the closing brackets are present, so a
    truncation repair does not apply, and guessing would invent arguments.
    """
    bad = '[{"content": "Open PR to main", "status": "pending", "priority"}]'
    got = coerce({"todos": bad})
    assert got["todos"] == bad, "malformed JSON must survive verbatim"


def test_type_mismatch_is_left_alone():
    """Schema wants an array; the JSON decodes to an object."""
    raw = json.dumps({"not": "an array"})
    got = coerce({"todos": raw})
    assert got["todos"] == raw


def test_plain_prose_is_not_parsed():
    got = coerce({"todos": "just some text"})
    assert got["todos"] == "just some text"


def test_number_like_string_is_not_parsed():
    """Only [ and { open a structure — '123' must not become an int."""
    got = coerce({"todos": "123"})
    assert got["todos"] == "123"


def test_empty_string_is_left_alone():
    got = coerce({"todos": ""})
    assert got["todos"] == ""


def test_already_correct_type_is_untouched():
    got = coerce({"todos": TODOS})
    assert got["todos"] == TODOS


def test_string_parameter_is_not_parsed():
    """A string-typed field holding JSON text stays a string."""
    got = coerce({"note": json.dumps(TODOS)})
    assert got["note"] == json.dumps(TODOS)


def test_unknown_key_is_untouched():
    got = coerce({"mystery": json.dumps(TODOS)})
    assert got["mystery"] == json.dumps(TODOS)


# --------------------------------------- the pre-existing direction (#regression)


def test_structure_for_string_schema_is_still_stringified():
    got = coerce({"note": {"a": 1}})
    assert isinstance(got["note"], str)
    assert json.loads(got["note"]) == {"a": 1}


def test_no_tools_is_a_passthrough():
    raw = json.dumps({"todos": json.dumps(TODOS)})
    assert srv._coerce_tool_arguments(raw, "todowrite", None) == raw


def test_unknown_tool_is_a_passthrough():
    raw = json.dumps({"todos": json.dumps(TODOS)})
    assert srv._coerce_tool_arguments(raw, "nosuchtool", _tool()) == raw


def test_non_json_arguments_are_a_passthrough():
    assert srv._coerce_tool_arguments("not json", "todowrite", _tool()) == "not json"


def test_idempotent():
    once = srv._coerce_tool_arguments(
        json.dumps({"todos": json.dumps(TODOS)}), "todowrite", _tool()
    )
    twice = srv._coerce_tool_arguments(once, "todowrite", _tool())
    assert json.loads(twice)["todos"] == TODOS


@pytest.mark.parametrize("depth", [1, 3, 8])
def test_nested_structures_survive_exactly(depth):
    payload = {"level": 0}
    for i in range(1, depth):
        payload = {"level": i, "child": payload}
    got = coerce({"todos": json.dumps([payload])}, tools=_tool())
    assert got["todos"] == [payload]
