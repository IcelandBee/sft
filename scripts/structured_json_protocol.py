#!/usr/bin/env python3
"""Constants for the format-controlled full-JSON generation protocol."""

from __future__ import annotations


JSON_CATEGORY = r'"[^"\\\n\r]{1,40}"'
JSON_REASON = r'"[^"\\\n\r]{1,80}"'
GOOD_PAYLOAD = r'\{"decision":"GOOD","categories":\[\],"reasons":\[\]\}'
BAD_PAYLOAD = (
    r'\{"decision":"BAD","categories":\['
    + JSON_CATEGORY
    + r'(,'
    + JSON_CATEGORY
    + r'){0,2}\],"reasons":\['
    + JSON_REASON
    + r'(,'
    + JSON_REASON
    + r'){0,2}\]\}'
)
STRUCTURED_OUTPUTS_REGEX = f"({GOOD_PAYLOAD}|{BAD_PAYLOAD})"


if __name__ == "__main__":
    print(STRUCTURED_OUTPUTS_REGEX)
