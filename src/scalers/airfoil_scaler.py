import numpy as np


# --- Helper Class for Normalization ---
class AirfoilScaler:
    def __init__(self):
        self.w_min = None
        self.w_max = None
        self.p_min = None
        self.p_max = None
        self.weights_per_surface = None

    def fit(self, weights, params):
        if weights.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected an even number of weight columns, got {weights.shape[1]}."
            )

        self.w_min = np.min(weights, axis=0)
        self.w_max = np.max(weights, axis=0)
        self.p_min = np.min(params, axis=0)
        self.p_max = np.max(params, axis=0)
        self.weights_per_surface = weights.shape[1] // 2

        # Safety for constant columns
        self.w_max[self.w_max == self.w_min] += 1e-6
        self.p_max[self.p_max == self.p_min] += 1e-6

    def transform(self, weights, params):
        if (
            self.w_min is None
            or self.w_max is None
            or self.p_min is None
            or self.p_max is None
        ):
            raise RuntimeError(
                "Scaler must be fitted before calling transform(). Call fit() first."
            )

        # Scale Weights to [-1, 1]
        w_scaled = 2 * (weights - self.w_min) / (self.w_max - self.w_min) - 1

        # Scale Parameters to [-1, 1]
        p_scaled = 2 * (params - self.p_min) / (self.p_max - self.p_min) - 1

        return np.concatenate([w_scaled, p_scaled], axis=1).astype(np.float32)

    def inverse_transform(self, weights, params):
        """
        Accepts normalized weights and params, returns real physical values.
        Handles both flat (B, 24) and structured (B, 2, 12) weight inputs.
        """
        if (
            self.w_min is None
            or self.w_max is None
            or self.p_min is None
            or self.p_max is None
        ):
            raise RuntimeError(
                "Scaler must be fitted before calling inverse_transform(). Call fit() first."
            )

        # Handle Weights Shape
        # If input is (Batch, 2, 12), flatten it to (Batch, 24)
        if weights.ndim == 3:
            weights = weights.reshape(weights.shape[0], -1)

        # Denormalize Weights from [-1, 1]
        w_orig = (weights + 1) / 2 * (self.w_max - self.w_min) + self.w_min

        # Denormalize Params from [-1, 1]
        p_orig = (params + 1) / 2 * (self.p_max - self.p_min) + self.p_min

        if self.weights_per_surface is None:
            raise RuntimeError(
                "Scaler weight metadata is missing. Call fit() before inverse_transform()."
            )

        # Reshape weights back to (Batch, 2, weights_per_surface) for easy plotting
        w_orig = w_orig.reshape(-1, 2, self.weights_per_surface)

        return w_orig, p_orig


# =============================================================================
# USAGE REFERENCE
# =============================================================================
#
# Example workflow for using AirfoilScaler:
#
# ─── Step 1: Prepare Raw Data ─────────────────────────────────────────────
# raw_data_matrix = np.stack(airfoil_data)       # Shape: (N_samples, 26)
# raw_weights = raw_data_matrix[:, :-2]         # First 24 columns
# raw_params = raw_data_matrix[:, -2:]           # Last 2 columns
#
# ─── Step 2: Initialize and Fit Scaler ────────────────────────────────────
# scaler = AirfoilScaler()
# scaler.fit(raw_weights, raw_params)
# print(f"Max Weight Value: {np.max(scaler.w_max)}")
# print(f"Max Param Value: {np.max(scaler.p_max)}")
#
# ─── Step 3: Transform Data (Normalization) ────────────────────────────────
# normalized_data = scaler.transform(raw_weights, raw_params)
#
# ─── Step 4: Create TensorFlow Dataset ─────────────────────────────────────
# train_dataset = tf.data.Dataset.from_tensor_slices(normalized_data)
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
# print(f"Dataset loaded and normalized: {len(normalized_data)} samples")
# print(f"Data Range Check -> Min: {normalized_data.min():.2f}, "
#       f"Max: {normalized_data.max():.2f}")
#
# =============================================================================
