import unittest
import numpy as np
from gareus.mbar_analysis.ladder import assert_lambda_sources_agree


class LambdaDisagreementTests(unittest.TestCase):
    def test_all_zero_registry_with_multi_valued_samples_raises(self):
        state_lambdas = np.zeros(112)
        per_sample = np.repeat([0.0, 0.5, 1.0], 100)
        with self.assertRaises(ValueError) as ctx:
            assert_lambda_sources_agree(state_lambdas, per_sample)
        self.assertIn("state_registry", str(ctx.exception))

    def test_genuine_non_ladder_run_is_silent(self):
        assert_lambda_sources_agree(np.zeros(8), np.zeros(500))

    def test_agreeing_ladder_run_is_silent(self):
        assert_lambda_sources_agree(np.array([0.0, 0.5, 1.0]), np.repeat([0.0, 0.5, 1.0], 10))

    def test_absent_per_sample_lambda_cannot_contradict(self):
        assert_lambda_sources_agree(np.zeros(8), None)
        assert_lambda_sources_agree(np.zeros(8), np.full(50, np.nan))


if __name__ == "__main__":
    unittest.main()
