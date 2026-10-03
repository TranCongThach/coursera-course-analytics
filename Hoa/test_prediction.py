"""Các bất biến quan trọng của phần ước lượng enrollment."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Support both `python -m Hoa.test_prediction` and
# `python Hoa/test_prediction.py` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Hoa import prediction as pred


class EnrollmentPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fact = pred.load_fact(pred.ENROLLMENT_COLUMNS)
        cls.model = pred.build_enrollment_model(cls.fact)

    def test_split_and_metrics_are_reproducible(self):
        other = pred.build_enrollment_model(self.fact)
        self.assertEqual(
            self.model.train_rows + self.model.calibration_rows + self.model.test_rows,
            self.model.eligible_rows,
        )
        np.testing.assert_allclose(self.model.coefficients, other.coefficients)
        self.assertEqual(self.model.metrics, other.metrics)
        self.assertGreater(self.model.eligible_rows, 100)
        self.assertLessEqual(self.model.eligible_rows, self.model.total_rows)

    def test_prediction_interval_and_inputs(self):
        values, warnings = self.model.estimate(1000, 4.6, 20, 6, "Beginner")
        self.assertLess(values["lower"], values["estimate"])
        self.assertLess(values["estimate"], values["upper"])
        self.assertEqual(warnings, [])
        with self.assertRaises(ValueError):
            self.model.estimate(0, 4.6, 20, 6, "Beginner")
        with self.assertRaises(ValueError):
            self.model.estimate(1000, 4.6, 20, -1, "Beginner")
        self.assertTrue(self.model.estimate(1_000_000_000, 4.6, 20, 6, "Beginner")[1])

    def test_target_derived_columns_are_excluded(self):
        self.assertFalse({"enrolled_percentile", "popularity_score", "review_to_enrollment_ratio"}
                         .intersection(pred.ENROLLMENT_COLUMNS))

    def test_course_trend_rejects_invalid_log_input(self):
        fact = pred.load_fact(["hours_to_complete", "enrolled_num"])
        with self.assertRaises(ValueError):
            pred.build_course_trend(fact, x_new=[0], log_x=True)
        with self.assertRaises(ValueError):
            pred.parse_number_list("10, abc")

    def test_internet_forecast_does_not_hide_out_of_range_prediction(self):
        years = np.arange(2000, 2024)
        history = pd.DataFrame({
            "country": "Example", "year": years,
            "internet_usage_pct": (years - 2000) * 4.0,
        })
        result = pred.build_internet_forecast("Example", history, horizon=5)
        self.assertGreater(result.table["forecast_pct"].max(), 100)
        self.assertTrue(any("[0, 100]" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
