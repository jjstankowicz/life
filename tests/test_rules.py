import unittest

from life import Rules


class TestRules(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_rules(self):
        rules = Rules()
        error_str = f"Expected die_when_under 2; got {rules.die_when_under}"
        self.assertEqual(rules.die_when_under, 2, error_str)
        error_str = f"Expected die_when_over 3; got {rules.die_when_over}"
        self.assertEqual(rules.die_when_over, 3, error_str)
        error_str = f"Expected dead_reproduction 3; got {rules.dead_reproduction}"
        self.assertEqual(rules.dead_reproduction, 3, error_str)
