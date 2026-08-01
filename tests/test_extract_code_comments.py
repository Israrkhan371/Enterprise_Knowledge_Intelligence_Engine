"""
Tests for app/ingestion/loaders.py::extract_code_comments().

Regression coverage for a real quality issue found via manual testing:
running entity extraction on raw source code (not just comments) produced
noise like "GenerationError", "IGNORECASE", "Delete", "Fallback" being
misread as named entities - class names, exception identifiers, and other
code tokens look enough like proper nouns to fool general-purpose NER.

extract_code_comments() pulls out only docstrings and comments, which is
where genuine human-language content (real names, real technology
references) actually lives in source code.
"""
from app.ingestion.loaders import extract_code_comments


def test_extracts_python_triple_quoted_docstring():
    code = '''
def foo():
    """
    This module was written by Aisha Rahman for the Ezitech platform team.
    """
    return 1
'''
    result = extract_code_comments(code)
    assert "Aisha Rahman" in result
    assert "def foo" not in result
    assert "return 1" not in result


def test_extracts_hash_line_comments():
    code = (
        "# Built by Farhan Chowdhury on the Retrieval team\n"
        "import chromadb\n"
        "x = 1  # not a real comment about people\n"
    )
    result = extract_code_comments(code)
    assert "Farhan Chowdhury" in result
    assert "not a real comment about people" in result
    assert "import chromadb" not in result
    assert "x = 1" not in result


def test_extracts_double_slash_line_comments():
    code = (
        "// Maintained by the Ezitech DevOps team\n"
        "const x = 1;\n"
    )
    result = extract_code_comments(code)
    assert "Maintained by the Ezitech DevOps team" in result
    assert "const x = 1" not in result


def test_extracts_block_comments():
    code = "/* Reviewed by Marcus Webb before merge */\nfunction f() {}\n"
    result = extract_code_comments(code)
    assert "Reviewed by Marcus Webb before merge" in result
    assert "function f" not in result


def test_returns_empty_string_for_code_with_no_comments():
    code = "def add(a, b):\n    return a + b\n"
    result = extract_code_comments(code)
    assert result == ""


def test_returns_empty_string_for_empty_input():
    assert extract_code_comments("") == ""


def test_handles_mixed_comment_styles_in_one_file():
    code = (
        '"""Module docstring mentioning Priya Chandrasekaran."""\n'
        "# Line comment mentioning Marcus Webb\n"
        "import os\n"
        "def f():\n"
        "    pass  # inline comment mentioning Sarah Lindqvist\n"
    )
    result = extract_code_comments(code)
    assert "Priya Chandrasekaran" in result
    assert "Marcus Webb" in result
    assert "Sarah Lindqvist" in result
    assert "import os" not in result
    assert "def f" not in result
