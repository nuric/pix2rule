"""Unit test cases for parameter scheduling functions."""
import unittest

from . import schedules


class TestDelayedExponentialDecay(unittest.TestCase):
    """Test cases for delayed exponential decay scheduler."""

    def test_initial_value(self):
        """Returns initial value on the first call."""
        sched = schedules.DelayedExponentialDecay(
            initial_learning_rate=0.42, decay_steps=1, decay_rate=0.9
        )
        new_value = sched(0)
        self.assertEqual(new_value, 0.42)

    def test_no_delay(self):
        """Returns regular decay if there is no delay."""
        sched = schedules.DelayedExponentialDecay(
            initial_learning_rate=1.00, decay_steps=1, decay_rate=0.9
        )
        new_value = sched(1)
        self.assertEqual(new_value, 1.00 * 0.9)

    def test_delayed_initial_value(self):
        """Returns decayed value after the specified delay time."""
        sched = schedules.DelayedExponentialDecay(
            initial_learning_rate=0.42, decay_steps=1, decay_rate=0.9, delay=10
        )
        new_value = sched(1)
        self.assertEqual(new_value, 0.42)

    def test_delayed_decay_value(self):
        """Returns decay after the specified delay."""
        sched = schedules.DelayedExponentialDecay(
            initial_learning_rate=1.00, decay_steps=1, decay_rate=0.9, delay=10
        )
        new_value = sched(11)
        self.assertEqual(new_value, 1.00 * 0.9)
