import json
import unittest

from scripts.evaluate_e1_dev import parse_prediction, validate_payload
from scripts.run_constrained_binary_dev import (
    BAD_PAYLOAD,
    GOOD_PAYLOAD,
    NON_THINKING_PREFIX,
    BinaryTokenTrie,
    ConstrainedBinaryError,
    _result_row,
)


class BinaryTokenTrieTests(unittest.TestCase):
    def test_follows_shared_prefix_then_branches_and_ends_with_eos(self):
        trie = BinaryTokenTrie([[1, 2, 3], [1, 2, 4, 5]], eos_token_id=9)

        self.assertEqual(trie.next_tokens([]), [1])
        self.assertEqual(trie.next_tokens([1]), [2])
        self.assertEqual(trie.next_tokens([1, 2]), [3, 4])
        self.assertEqual(trie.next_tokens([1, 2, 3]), [9])
        self.assertEqual(trie.next_tokens([1, 2, 4]), [5])
        self.assertEqual(trie.next_tokens([1, 2, 4, 5]), [9])

    def test_rejects_duplicate_prefix_and_out_of_trie_candidates(self):
        with self.assertRaises(ConstrainedBinaryError):
            BinaryTokenTrie([[1, 2], [1, 2]], eos_token_id=9)
        with self.assertRaises(ConstrainedBinaryError):
            BinaryTokenTrie([[1, 2], [1, 2, 3]], eos_token_id=9)
        trie = BinaryTokenTrie([[1, 2], [1, 3]], eos_token_id=9)
        with self.assertRaises(ConstrainedBinaryError):
            trie.next_tokens([4])

    def test_callback_uses_only_tokens_after_bound_prompt(self):
        trie = BinaryTokenTrie([[4, 5], [4, 6]], eos_token_id=9)
        trie.bind_prompt_length(3)

        class FakeIds:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        self.assertEqual(trie.prefix_allowed_tokens_fn(0, FakeIds([8, 8, 8])), [4])
        self.assertEqual(trie.prefix_allowed_tokens_fn(0, FakeIds([8, 8, 8, 4])), [5, 6])


class BinaryProtocolTests(unittest.TestCase):
    def test_candidate_payloads_are_valid_under_existing_strict_schema(self):
        for payload in (GOOD_PAYLOAD, BAD_PAYLOAD):
            with self.subTest(payload=payload):
                value = json.loads(payload)
                self.assertIsNone(validate_payload(value))
                parsed = parse_prediction(NON_THINKING_PREFIX + payload)
                self.assertTrue(parsed["schema_valid"])

    def test_result_row_is_compatible_with_existing_evaluator_envelope(self):
        source = {
            "images": ["/image.jpg"],
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": GOOD_PAYLOAD},
            ],
        }
        response = NON_THINKING_PREFIX + BAD_PAYLOAD

        row = _result_row(source, response, None)

        self.assertEqual(row["response"], response)
        self.assertEqual(row["labels"], GOOD_PAYLOAD)
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": response})
        self.assertEqual(row["images"], [{"bytes": None, "path": "/image.jpg"}])


if __name__ == "__main__":
    unittest.main()
