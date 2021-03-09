"""Slot attention based auto encoder model."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

import configlib
from configlib import config as C
from reportlib import ReportLayer

from components.util_layers import SpacialFlatten, SpacialBroadcast
from components.slot_attention import SlotAttention, SoftPositionEmbed

import components.inputlayers.image

import utils.callbacks
import utils.factory
import utils.schedules

# ---------------------------
# Setup configurable parameters of the model
add_argument = configlib.add_group(
    "Slot attention auto encoder parameters", prefix="slotae"
)
# ---
# Image layer parameters
configlib.add_arguments_dict(
    add_argument, components.inputlayers.image.configurable, prefix="image"
)
# ---------------------------


class RecombineStackedImage(L.Layer):
    """Recombines a broadcasted image reconstruction."""

    def __init__(self, num_channels: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels

    def call(self, inputs: tf.Tensor, **kwargs):
        """Unstack, split and recombine reconstructed image."""
        # inputs (B, S, E), (B*S, W, H, 4)
        # We pass both inputs to reshape B*S into S
        new_shape = tf.concat([tf.shape(inputs[0])[:2], tf.shape(inputs[1])[1:]], 0)
        # (B, S, W, H, 4)
        unstacked = tf.reshape(inputs[1], new_shape)  # (B, S, W, H, 4)
        channels, masks = tf.split(
            unstacked, [self.num_channels, 1], -1
        )  # [(B, S, W, H, 3), (B, S, W, H, 1)]
        masks = tf.nn.softmax(masks, axis=1)  # (B, S, W, H, 1)
        reconstruction = tf.reduce_sum(channels * masks, axis=1)  # (B, W, H, 3)
        return {
            "reconstruction": reconstruction,
            "recon_masks": masks,
            "slot_recon": channels,
        }


def process_image(image: tf.Tensor, _: Dict[str, Any]) -> tf.Tensor:
    """Process given image input to extract facts."""
    # image (B, W, H, C)
    # ---------------------------
    # Process the images
    image_layer = utils.factory.get_and_init(
        components.inputlayers.image, C, "slotae_image_", name="image_layer"
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    raw_objects = SoftPositionEmbed(32, [12, 12])(raw_objects)  # (B, W, H, E)
    raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
    return raw_objects


def build_model(  # pylint: disable=too-many-locals
    task_description: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the DNF trainable model."""
    # ---------------------------
    # Setup and process inputs
    slotae_inputs = utils.factory.create_input_layers(
        task_description, {"image": process_image}
    )
    hidden_size = 32
    # ---------------------------
    object_mlp = tf.keras.Sequential(
        [L.Dense(hidden_size, activation="relu"), L.Dense(hidden_size)],
        name="feedforward",
    )
    norm_objects = L.LayerNormalization()(
        slotae_inputs["processed"]["image"]
    )  # (B, O, E)
    objects = object_mlp(norm_objects)  # (B, O, E)
    slots_dict = SlotAttention(
        num_iterations=3, num_slots=5, slot_size=hidden_size, mlp_hidden_size=64
    )(objects)
    # slots_dict['slots'] # (B, S, E)
    slots_dict = ReportLayer()(slots_dict)
    # ---------------------------
    # Decode Image
    decoder_initial_res = [3, 3]
    decoded = SpacialBroadcast(decoder_initial_res)(
        slots_dict["slots"]
    )  # (B*S, W, H, E)
    decoded = SoftPositionEmbed(hidden_size, decoder_initial_res)(
        decoded
    )  # (B*S, W, H, E)
    decoder_cnn = tf.keras.Sequential(
        [
            L.Conv2DTranspose(
                hidden_size, 5, strides=2, padding="SAME", activation="relu"
            ),
            L.Conv2DTranspose(
                hidden_size, 5, strides=2, padding="SAME", activation="relu"
            ),
            L.Conv2DTranspose(
                hidden_size, 5, strides=1, padding="SAME", activation="relu"
            ),
            L.Conv2DTranspose(4, 5, strides=1, padding="SAME", activation=None),
        ],
        name="decoder_cnn",
    )
    decoded = decoder_cnn(decoded)  # (B*S, W, H, 4)
    # ---------------------------
    # Recombine image for final prediction
    recon_dict = RecombineStackedImage(num_channels=3)(
        [slots_dict["slots"], decoded]
    )  # (B, W, H, C)
    recon_dict = ReportLayer()(recon_dict)
    # ---------------------------
    # Create model with given inputs and outputs
    model = tf.keras.Model(
        inputs=slotae_inputs["input_layers"],
        outputs={"image": recon_dict["reconstruction"]},
        name="slotae_model",
    )
    # ---------------------------
    # Compile model for training
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    # ---------------------------
    return {"model": model, "loss": loss, "metrics": metrics}
