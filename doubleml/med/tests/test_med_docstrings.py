import inspect
import re

import pytest

from doubleml import DoubleMLMED, DoubleMLMEDData, DoubleMLMEDS
from doubleml.med.datasets import make_med_data

_SECTIONS = {"Parameters", "Returns", "Raises", "Examples", "Notes", "References", "Attributes"}


def _public_docstrings():
    docstrings = {"make_med_data": inspect.getdoc(make_med_data)}
    for model_class in (DoubleMLMED, DoubleMLMEDS, DoubleMLMEDData):
        docstrings[model_class.__name__] = inspect.getdoc(model_class)
        for name, member in inspect.getmembers(model_class):
            is_public = not name.startswith("_") or name == "__init__"
            if is_public and (inspect.isfunction(member) or isinstance(member, property)):
                docstrings[f"{model_class.__name__}.{name}"] = inspect.getdoc(member)
    return {label: doc for label, doc in docstrings.items() if doc}


@pytest.mark.ci
@pytest.mark.parametrize("label, docstring", list(_public_docstrings().items()), ids=list(_public_docstrings()))
def test_docstring_sections_are_consistently_indented(label, docstring):
    """After removing the common indentation, headings, underlines, parameter entries and doctests start at column 0."""
    lines = docstring.split("\n")
    misindented = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or line == stripped:
            continue
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        is_heading = stripped in _SECTIONS and re.fullmatch(r"\s*-{3,}\s*", next_line) is not None
        is_underline = re.fullmatch(r"-{3,}", stripped) is not None
        is_doctest = stripped.startswith(">>>")
        is_entry = re.match(r"^[A-Za-z_][A-Za-z_0-9, ]* : ", stripped) is not None and not line.startswith("    ")
        if is_heading or is_underline or is_doctest or is_entry:
            misindented.append(f"line {i + 1}: {line!r}")

    assert not misindented, f"{label} has misindented lines: " + "; ".join(misindented[:3])
