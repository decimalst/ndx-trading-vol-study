"""Inspect XML structure while withholding all scalar source content."""

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


def inspect_xml_schema(raw: bytes) -> dict:
    if type(raw) is not bytes or not 0 < len(raw) <= 2 * 1024 * 1024:
        raise ValueError("nonempty XML bytes limited to 2 MiB required")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("XML must be valid UTF-8") from exc
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.I):
        raise ValueError("DTD and entity declarations prohibited")
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("invalid XML") from exc
    tags = Counter()
    attrs = defaultdict(set)
    todo = [(root, root.tag)]
    while todo:
        element, path = todo.pop()
        tags[path] += 1
        attrs[path].update(element.attrib.keys())
        todo.extend((child, path + "/" + child.tag) for child in element)
    return {
        "root_tag": root.tag,
        "tag_counts": dict(sorted(tags.items())),
        "attribute_names": {path: sorted(attrs[path]) for path in sorted(tags)},
    }
