import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression

def generate_perturbations(data, num_samples=500):
    """
    Generate perturbed versions of the input data.
    """
    perturbed_data = np.random.normal(data, 0.01, size=(num_samples, *data.shape))
    return perturbed_data

def lime_explanation(model, data, num_samples=500, pool_channels=True):
    """
    Explain a prediction using LIME for multivariate time series data.

    Args:
    - model: The TensorFlow model to explain.
    - data: The input data instance to explain, shaped as (1, time_steps, channels).
    - num_samples: Number of samples to generate for the explanation.
    - pool_channels: Whether to pool explanations across channels or not.

    Returns:
    - Coefficients indicating feature importance, shaped as (channels,) or a single value if pooled.
    """
    # Generate perturbed data samples
    perturbed_data = generate_perturbations(data, num_samples)
    original_shape = perturbed_data.shape[1:]

    # Predict using the original model
    predictions = model.predict(perturbed_data).flatten()

    # Fit a linear model for each channel if not pooling, else pool data
    if pool_channels:
        # Pooling perturbed data across channels
        perturbed_data_pooled = perturbed_data.reshape(num_samples, -1)
        linear_model = LinearRegression()
        linear_model.fit(perturbed_data_pooled, predictions)
        coefficients = linear_model.coef_.reshape(original_shape).mean(axis=0)  # Mean importance across all channels
    else:
        coefficients = np.zeros(original_shape[1])  # Channels
        for channel in range(original_shape[1]):
            # Selecting data for the current channel
            channel_data = perturbed_data[:, :, channel].reshape(num_samples, -1)
            linear_model = LinearRegression()
            linear_model.fit(channel_data, predictions)
            coefficients[channel] = linear_model.coef_.mean()

    return coefficients

# Example usage:
# model = ...  # Your 1D CNN model
# input_data = ...  # Your input data here, shaped (1, 2500, 12) for a single instance
# pred_index = None  # Optional: class to visualize (None for the highest prediction)

# For pooled explanation
# pooled_coefficients = lime_explanation(model, input_data, pool_channels=True)
# print("Pooled coefficients:", pooled_coefficients)

# For individual channel explanations
# channel_coefficients = lime_explanation(model, input_data, pool_channels=False)
# print("Channel-specific coefficients:", channel_coefficients)
