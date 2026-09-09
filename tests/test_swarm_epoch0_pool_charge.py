"""Epoch 0's sampling comes out of the same budget as every other epoch.

The swarm is real MD -- 174 members x 1 ns is not a rounding error against a
6000 ns campaign -- so it has to be billed, and billed exactly once no matter how
many walltime stops the epoch survives.
"""
import unittest

from gareus.adaptive_production import AdaptiveRuntimePool, _charge_swarm_to_pool


class SwarmPoolChargeTests(unittest.TestCase):
    def test_charges_the_nanoseconds_the_swarm_actually_ran(self):
        # The pool bills whole MD steps, so the charge lands within half a
        # timestep (~2e-6 ns at 3.5 fs) of the requested nanoseconds. Anything
        # tighter than that is asserting on floating-point noise, not accounting.
        pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
        _charge_swarm_to_pool(pool, 174.0, None)
        self.assertAlmostEqual(pool.used_ns, 174.0, places=4)
        self.assertAlmostEqual(pool.remaining_ns(), 6000.0 - 174.0, places=4)

    def test_the_charge_is_independent_of_the_pool_timestep(self):
        """consume() derives ns from steps x timestep, so the inversion must track it."""
        for dt in (2.0, 3.0, 3.5, 4.0):
            pool = AdaptiveRuntimePool(total_ns=1000.0, timestep_fs=dt)
            _charge_swarm_to_pool(pool, 174.0, None)
            self.assertAlmostEqual(pool.used_ns, 174.0, places=3, msg=f"timestep {dt}")

    def test_it_is_recorded_as_its_own_labelled_event(self):
        pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
        _charge_swarm_to_pool(pool, 50.0, None)
        self.assertEqual(len(pool.events), 1)
        self.assertEqual(pool.events[0]["label"], "epoch_000_swarm")
        self.assertEqual(pool.events[0]["kind"], "swarm_epoch0")

    def test_a_zero_or_absent_charge_is_a_no_op(self):
        pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
        _charge_swarm_to_pool(pool, 0.0, None)
        _charge_swarm_to_pool(None, 174.0, None)
        self.assertEqual(pool.used_ns, 0.0)
        self.assertEqual(pool.events, [])


if __name__ == "__main__":
    unittest.main()
