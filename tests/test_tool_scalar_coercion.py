# SPDX-License-Identifier: Apache-2.0
"""PATCHES.md #94 — scalar tool arguments must land on their declared type.

The XML-ish tool dialects carry EVERY ``<parameter=name>`` value as raw text,
so a ``number``/``integer``/``boolean`` field arrives as ``"440"`` / ``"true"``
just as an array arrived as a JSON string under #92. Observed live as
``SchemaError(Expected number | undefined, got "440" at ["offset"])``.
"""

import json

import pytest

import vllm_mlx.server as srv


def _tool(prop_type, name="read", key="offset"):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {key: {"type": prop_type}},
                },
            },
        }
    ]


def coerce(args, prop_type, key="offset", name="read"):
    tools = _tool(prop_type, name=name, key=key)
    out = srv._coerce_tool_arguments(json.dumps(args), name, tools)
    return json.loads(out)[key]


# ------------------------------------------------ the live failures


def test_integer_offset_from_string():
    """`read` with offset="440" — verbatim from the session audit."""
    got = coerce({"offset": "440"}, "integer")
    assert got == 440 and isinstance(got, int)


@pytest.mark.parametrize("raw", ["300000", "600000"])
def test_bash_timeout_from_string(raw):
    got = coerce({"timeout": raw}, "number", key="timeout", name="bash")
    assert got == int(raw)


# ------------------------------------------------ scalar coercion


def test_number_float():
    assert coerce({"offset": "1.5"}, "number") == 1.5


def test_negative_and_exponent():
    assert coerce({"offset": "-42"}, "integer") == -42
    assert coerce({"offset": "1e3"}, "number") == 1000.0


def test_boolean_true_and_false():
    assert coerce({"offset": "true"}, "boolean") is True
    assert coerce({"offset": "false"}, "boolean") is False


def test_zero_and_false_are_coerced_not_skipped():
    """Falsy values must not be mistaken for 'no coercion'."""
    assert coerce({"offset": "0"}, "integer") == 0
    assert coerce({"offset": "false"}, "boolean") is False


def test_whitespace_wrapped_scalar():
    assert coerce({"offset": "  440\n"}, "integer") == 440


def test_integral_float_accepted_for_integer():
    got = coerce({"offset": "4.0"}, "integer")
    assert got == 4 and isinstance(got, int)


# ------------------------------------------------ the guards


def test_bool_does_not_satisfy_integer():
    """bool is an int subclass in Python — the classic trap."""
    assert coerce({"offset": "true"}, "integer") == "true"


def test_int_does_not_satisfy_boolean():
    assert coerce({"offset": "1"}, "boolean") == "1"


def test_fractional_float_rejected_for_integer():
    assert coerce({"offset": "4.5"}, "integer") == "4.5"


def test_union_with_string_is_left_alone():
    """If a string is legal the text already IS the value; converting would
    change its meaning."""
    assert coerce({"offset": "440"}, ["string", "number"]) == "440"


def test_string_typed_field_untouched():
    assert coerce({"offset": "440"}, "string") == "440"


def test_non_numeric_text_untouched():
    assert coerce({"offset": "not a number"}, "integer") == "not a number"


def test_malformed_number_untouched():
    for raw in ["0x10", "007", "1,000", "44 40", ""]:
        assert coerce({"offset": raw}, "integer") == raw


def test_null_is_not_coerced():
    assert coerce({"offset": "null"}, "integer") == "null"


def test_already_numeric_untouched():
    assert coerce({"offset": 440}, "integer") == 440


# ------------------------------------------------ #92 must not regress


def test_array_still_coerced():
    todos = [{"content": "a"}]
    got = coerce({"offset": json.dumps(todos)}, "array")
    assert got == todos


def test_object_still_coerced():
    payload = {"a": 1}
    assert coerce({"offset": json.dumps(payload)}, "object") == payload


def test_malformed_array_still_left_verbatim():
    bad = '[{"a": 1, "b"}]'
    assert coerce({"offset": bad}, "array") == bad


def test_structure_for_string_schema_still_stringified():
    tools = _tool("string")
    out = srv._coerce_tool_arguments(json.dumps({"offset": {"a": 1}}), "read", tools)
    assert isinstance(json.loads(out)["offset"], str)
