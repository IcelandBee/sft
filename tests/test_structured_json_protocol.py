import re
import unittest

from scripts.structured_json_protocol import STRUCTURED_OUTPUTS_REGEX


class StructuredJsonProtocolTests(unittest.TestCase):
    def test_accepts_valid_compact_good_and_bad_payloads(self):
        valid = [
            '{"decision":"GOOD","categories":[],"reasons":[]}',
            '{"decision":"BAD","categories":["手部异常"],"reasons":["手指畸形"]}',
            '{"decision":"BAD","categories":["手部异常","人体结构异常"],"reasons":["手指畸形","多出手臂"]}',
        ]
        for payload in valid:
            with self.subTest(payload=payload):
                self.assertIsNotNone(re.fullmatch(STRUCTURED_OUTPUTS_REGEX, payload))

    def test_rejects_non_json_missing_fields_and_invalid_auxiliary_shapes(self):
        invalid = [
            'GOOD',
            '{"decision":"GOOD","categories":["手部异常"],"reasons":[]}',
            '{"decision":"BAD","categories":[],"reasons":[]}',
            '{"decision":"BAD","categories":["手部异常"],"reasons":[]}',
            '{"decision":"BAD","categories":["手部异常"],"reasons":["a","b","c","d"]}',
            ' {"decision":"GOOD","categories":[],"reasons":[]}',
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertIsNone(re.fullmatch(STRUCTURED_OUTPUTS_REGEX, payload))


if __name__ == "__main__":
    unittest.main()
