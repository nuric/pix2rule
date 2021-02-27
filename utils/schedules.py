"""Parameter schedules for models."""
import tensorflow as tf


class DelayedExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    """Delays the start of exponential decay by given number of epochs."""

    def __init__(self, *args, delay: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def __call__(self, step: int):
        """Perform delayed decay."""
        if step < self.delay:
            return self.initial_learning_rate
        return super().__call__(step - self.delay)

    def get_config(self):
        """Configuration for serilisation."""
        config = super().get_config()
        config.update(
            {
                "delay": self.delay,
            }
        )
        return config
