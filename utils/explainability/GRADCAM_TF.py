import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def compute_gradcam_heatmaps(input_model, input_data, conv_layer_name, pred_index=None, pool_channels=True):
    """
    Generate Grad-CAM heatmaps for a 1D CNN model.

    Args:
    - input_model: The TensorFlow/Keras model being interpreted.
    - input_data: The input data for which the explanation is sought, shaped as (1, time_steps, channels).
    - conv_layer_name: The name of the convolutional layer to use for Grad-CAM.
    - pred_index: The index of the class to explain. If None, the class with the highest prediction is used.
    - pool_channels: Whether to pool across channels to generate a single heatmap, or return individual heatmaps for each channel.

    Returns:
    - A numpy array containing the heatmap(s). This will be of shape (time_steps,) if pool_channels is True,
      or (time_steps, channels) if pool_channels is False.
    """
    # Create a model that outputs the target convolutional layer's output and the prediction
    grad_model = tf.keras.models.Model(
        [input_model.inputs],
        [input_model.get_layer(conv_layer_name).output, input_model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # Compute gradients with respect to the activations of the convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    if pool_channels:
        # Pool gradients across the channels, and then compute the weighted average
        pooled_grads = tf.reduce_mean(grads, axis=-1)
        conv_outputs = conv_outputs[0]
        heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap).numpy()
    else:
        # Compute the weighted average for each channel
        weights = tf.reduce_mean(grads, axis=1)
        conv_outputs = conv_outputs[0]
        heatmap = tf.einsum('ijk,ik->ij', conv_outputs, weights)
        heatmap = heatmap.numpy()

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# Example usage:
# model = ...  # Your 1D CNN model
# input_data = ...  # Your input data here, shaped (1, 2500, 12) for a single instance
# conv_layer_name = 'your_conv_layer_name'  # Name of your chosen conv layer
# pred_index = None  # Optional: class to visualize (None for the highest prediction)

# For pooled heatmap
# pooled_heatmap = compute_gradcam_heatmaps(model, input_data, conv_layer_name, pool_channels=True)
# plt.matshow(pooled_heatmap)
# plt.colorbar()
