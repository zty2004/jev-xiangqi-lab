import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "train" / "compare_choice_values.py"
spec = importlib.util.spec_from_file_location("compare_choice_values", MODULE)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class CompareChoiceValuesTests(unittest.TestCase):
    def test_weighted_metrics_ignore_zero_weight_targets(self):
        metrics = compare.weighted_metrics([(0.5, 1.0), (99.0, 0.0), (-0.25, 3.0)])
        self.assertEqual(metrics["positions"], 2)
        self.assertEqual(metrics["effectiveWeight"], 4.0)
        self.assertAlmostEqual(metrics["mae"], 0.3125)
        self.assertAlmostEqual(metrics["mse"], 0.109375)

    def test_interval_uses_empirical_middle_95_percent(self):
        values = list(range(2000))
        self.assertEqual(compare.interval(values), [50, 1949])


if __name__ == "__main__":
    unittest.main()
