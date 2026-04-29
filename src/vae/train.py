# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import random
import joblib

from pathlib import Path
from aerosandbox import Airfoil, KulfanAirfoil

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from src import vae
from src.vae import CSTVariationalAutoencoder
from src.plotting import plot_original_and_reconstruction
from src.utils import compute_vod_loss
from src.layers.airfoil_scaler import AirfoilScaler

from src.airfoil import airfoil_modifications

# ============================================================================
# CONFIGURATION AND RANDOM SEEDS
# ============================================================================

SEED = 42  # Fixed seed for reproducibility across all random operations
AIRFOILS_TO_PLOT = 9  # Number of validation airfoils to visualize during training
CHECKPOINT_EPOCHS = 10  # Save model and visualization every N epochs
VERBOSE = 1  # 0: silent, 1: print epoch info
DEV = False  # Development mode flag (set to False for full training with WandB logging)

# Set seeds for all libraries to ensure reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

EPOCHS = 500  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training
LATENT_DIM = 16  # Dimensionality of the latent space
NPV = 8  # Number of CST coefficients per surface (MUST match dataset generation)
LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
CLIPNORM = 1.0  # Gradient clipping norm to prevent exploding gradients
WARMUP_EPOCHS = 100  # Number of epochs for KL annealing warm-up
TARGET_BETA = 0.01  # Final weight for KL Divergence loss (reached after warmup)
SMOOTHNESS_WEIGHT = 0  # Weight for the smoothness loss term

HYPERPARAMETERS = {
    "epochs": EPOCHS,
    "latent_dim": LATENT_DIM,
    "learning_rate": LEARNING_RATE,
    "target_beta": TARGET_BETA,
    "warmup_epochs": WARMUP_EPOCHS,
    "batch_size": BATCH_SIZE,
    "clipnorm": CLIPNORM,
    "smoothness_weight": SMOOTHNESS_WEIGHT,
}

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
TIMESTRING = time.strftime("%Y%m%d-%H%M%S")

# ============================================================================
# DATASET LOADING AND PREPARATION
# ============================================================================

# Load the Kulfan parameter dataset
train_dataset_path = (
    Path(PROJECT_PATH) / "data" / "processed" / "train_kulfan_dataset_75.json"
)
print("\n" + "=" * 70)
print("LOADING TRAINING DATASET")
print("=" * 70)
airfoil_dataset = pd.read_json(train_dataset_path)
print(f"✓ Loaded {len(airfoil_dataset)} training samples")

# Convert coordinate strings to numpy arrays
airfoil_dataset["coordinates"] = airfoil_dataset["coordinates"].apply(
    lambda coords: np.array(coords)
)

# Extract and concatenate Kulfan parameters:
# [lower_weights(12), upper_weights(12), TE_thickness(1), leading_edge_weight(1)]
# Total: 26 parameters per airfoil
airfoil_data = (
    airfoil_dataset["kulfan_parameters"]
    .apply(
        lambda p: np.concatenate(
            [
                p["upper_weights"],  # Upper surface CST weights
                p["lower_weights"],  # Lower surface CST weights
                [p["TE_thickness"]],  # Trailing edge thickness
                [p["leading_edge_weight"]],  # Leading edge weight
            ],
            axis=0,
        )
    )
    .to_numpy()
)

airfoil_data = np.stack(airfoil_data, axis=0).astype(np.float32)

# Split weights and parameters for normalization
raw_weights = airfoil_data[:, :-2]  # All CST coefficients (24 total)
raw_params = airfoil_data[:, -2:]  # TE thickness and leading edge weight
print(f"✓ Data shape: Weights {raw_weights.shape} | Params {raw_params.shape}")

# Fit scaler on clean data
print("✓ Fitting scaler to data...")
scaler = AirfoilScaler()
scaler.fit(raw_weights, raw_params)

print(f"  Weight range: ±{np.max(scaler.w_max):.6f}")
print(f"  Param range:  ±{np.max(scaler.p_max):.6f}")

# Normalize the data to [-1, 1] range
normalized_data = scaler.transform(raw_weights, raw_params)

# Create TensorFlow dataset pipeline with shuffling and batching
train_dataset = tf.data.Dataset.from_tensor_slices(normalized_data)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
print(f"\n✓ Dataset normalized: {len(normalized_data)} samples")
print(
    f"  Batch size: {BATCH_SIZE} | Total batches: {len(normalized_data) // BATCH_SIZE}"
)
print(f"  Data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

# ============================================================================
# VALIDATION AIRFOILS PREPARATION
# ============================================================================

validation_dataset_path = (
    Path(PROJECT_PATH) / "data" / "processed" / "val_kulfan_dataset_75.json"
)
print("\n" + "=" * 70)
print("LOADING VALIDATION DATASET")
print("=" * 70)
validation_airfoil_dataset = pd.read_json(validation_dataset_path)
print(f"✓ Validation dataset loaded: {len(validation_airfoil_dataset)} samples")

# Convert coordinate strings to numpy arrays
validation_airfoil_dataset["coordinates"] = validation_airfoil_dataset[
    "coordinates"
].apply(lambda coords: np.array(coords))

# Select first N airfoils for validation visualization during training
validation_airfoils_sample = validation_airfoil_dataset.iloc[
    :AIRFOILS_TO_PLOT
].reset_index(drop=True)
print(
    f"✓ Selected {len(validation_airfoils_sample)} airfoils for validation visualization"
)
# Create Airfoil objects for plotting reference
validation_airfoils = [
    Airfoil(coordinates=af["coordinates"], name=af["airfoil_name"])
    for af in validation_airfoils_sample.to_dict(orient="records")
]

# Extract Kulfan parameters from validation airfoils
validation_input = (
    validation_airfoils_sample["kulfan_parameters"]
    .apply(
        lambda p: np.concatenate(
            [
                p["upper_weights"],  # Upper surface weights
                p["lower_weights"],  # Lower surface weights
                [p["TE_thickness"]],  # Trailing edge thickness
                [p["leading_edge_weight"]],  # Leading edge weight
            ],
            axis=0,
        )
    )
    .to_list()
)

# Convert to tensor and normalize using the fitted scaler
validation_input = tf.convert_to_tensor(validation_input, dtype=tf.float32)
weights = validation_input[:, :24]  # Extract weights
params = validation_input[:, 24:]  # Extract parameters
validation_input = scaler.transform(weights, params)  # Normalize to [-1, 1]

# ============================================================================
# MODEL, OPTIMIZER, AND LOSS INITIALIZATION
# ============================================================================
# Initialize the VAE model
vae = CSTVariationalAutoencoder(scaler, npv=NPV, latent_dim=LATENT_DIM)

# Variables for early stopping and learning rate scheduling (optional)
best_loss = float("inf")
wait = 0
current_lr = LEARNING_RATE

# Adam optimizer with gradient clipping to prevent exploding gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, clipnorm=CLIPNORM)

# Reconstruction loss function (MAE is more robust than MSE for this task)
# reconstruction_loss = tf.keras.losses.MeanSquaredError()  # Alternative: MSE
reconstruction_loss = tf.keras.losses.MeanAbsoluteError()


# ============================================================================
# TRAINING STEP FUNCTION
# ============================================================================
@tf.function
def train_step(data, beta):
    """
    Runs one training step with Sum-Squared Error reconstruction loss.

    Args:
        data: Normalized airfoil parameters (Batch, 26)
        beta: Current KL divergence weight for annealing

    Returns:
        total_loss: Combined reconstruction + KL loss
        reco_loss: Reconstruction loss only
        kl_loss: KL divergence loss only
    """
    with tf.GradientTape() as tape:
        # Forward pass: training=True enables stochastic sampling from latent distribution
        reconstruction = vae(data, training=True)
        pred_weights, pred_params = reconstruction

        # Split ground truth: data shape is (Batch, 26)
        true_weights, true_params = tf.split(data, [2 * NPV, 2], axis=1)

        # Flatten predicted weights from (Batch, 2, 12) to (Batch, 24) for loss computation
        pred_weights_flat = tf.reshape(pred_weights, [-1, 2 * NPV])

        # Compute reconstruction losses using Sum of Squared Errors
        loss_weights = tf.reduce_mean(
            tf.reduce_sum(tf.square(true_weights - pred_weights_flat), axis=1)
        )
        loss_params = tf.reduce_mean(
            tf.reduce_sum(tf.square(true_params - pred_params), axis=1)
        )

        # Combined reconstruction loss
        reco_loss = loss_weights + loss_params

        # KL divergence loss (computed via self.add_loss in the model)
        kl_loss = sum(vae.losses)

        # Calculating predicted "normalized" coordinates
        pred_coords_norm = vae.decoder.cst_transform(pred_weights, pred_params)

        # Extract Y coordinates
        y_pred_coords_norm = pred_coords_norm[:, :, 1]

        # VOD (Smoothness) loss
        vod_loss = compute_vod_loss(y_pred_coords_norm)

        # Total loss: reconstruction + annealed KL divergence + vod loss
        total_loss = reco_loss + (beta * kl_loss) + (SMOOTHNESS_WEIGHT * vod_loss)

    # Backpropagation
    grads = tape.gradient(total_loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    return total_loss, reco_loss, kl_loss, vod_loss


# ============================================================================
# WEIGHTS & BIASES INITIALIZATION
# ============================================================================
# Initialize WandB for experiment tracking and logging
if not DEV:
    wandb.init(
        project="CSTVAE",
        config=HYPERPARAMETERS,
        name=f"VAE_{TIMESTRING}",  # Unique run name with timestamp
        notes="Dense Arch + Linear Output + Sum Loss + Scaler",
    )
    print("\n✓ WandB initialized for experiment tracking")
else:
    print("\n⚠ Development mode enabled - WandB logging disabled")

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
# Create timestamped directories for model checkpoints and visualization images
models_path = Path(PROJECT_PATH) / "models" / "cstvae" / TIMESTRING / "weights"
scaler_path = Path(PROJECT_PATH) / "models" / "cstvae" / TIMESTRING / "scaler"
images_path = Path(PROJECT_PATH) / "models" / "cstvae" / TIMESTRING / "images"
os.makedirs(models_path, exist_ok=True)  # Create model save directory
os.makedirs(scaler_path, exist_ok=True)  # Create scaler directory
os.makedirs(images_path, exist_ok=True)  # Create visualization output directory
print("\n" + "=" * 70)
print("OUTPUT CONFIGURATION")
print("=" * 70)
print(f"✓ Model checkpoints: {models_path}")
print(f"✓ Visualizations:    {images_path}")

# ============================================================================
# PREPARE FULL VALIDATION SET FOR METRICS
# ============================================================================
print("Preparing full validation tensor for metrics...")

# Extract ALL validation vectors
val_full_vectors = (
    validation_airfoil_dataset["kulfan_parameters"]
    .apply(
        lambda p: np.concatenate(
            [
                p["upper_weights"],
                p["lower_weights"],
                [p["TE_thickness"]],
                [p["leading_edge_weight"]],
            ],
            axis=0,
        )
    )
    .to_list()
)

# Convert to Tensor
val_full_tensor = tf.convert_to_tensor(val_full_vectors, dtype=tf.float32)

# Normalize (using the same scaler fitted on training data)
val_full_normalized = scaler.transform(val_full_tensor[:, :24], val_full_tensor[:, 24:])

print(f"✓ Full validation set ready: {val_full_normalized.shape}")

# ============================================================================
# PREPARE "GROUND TRUTH" GEOMETRY FOR METRICS
# ============================================================================
print("Generating 'Ground Truth' geometry for geometric metrics...")

# Get PHYSICAL values (Denormalized/Real) from validation dataset
true_weights_tensor = val_full_tensor[:, :24]
true_params_tensor = val_full_tensor[:, 24:]

# Generate real X,Y coordinates using the CST Layer
# Resulting Shape: (N_Samples, N_Points, 2)
true_weights_reshaped = tf.reshape(true_weights_tensor, [-1, 2, 12])
true_geo_coords = vae.decoder.cst_transform(true_weights_reshaped, true_params_tensor)

print(f"✓ Target geometry ready. Shape: {true_geo_coords.shape}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"  Epochs:         {EPOCHS}")
print(f"  Batch Size:     {BATCH_SIZE}")
print(f"  Latent Dim:     {LATENT_DIM}")
print(f"  Learning Rate:  {LEARNING_RATE}")
print(f"  Warmup Epochs:  {WARMUP_EPOCHS}")
print(f"  Target Beta:    {TARGET_BETA}")
print(f"  Dev Mode:       {'ON (WandB disabled)' if DEV else 'OFF (WandB enabled)'}")
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70 + "\n")

start_time = time.time()

# Save the Scaler
scaler_fn = f"scaler.pkl"
joblib.dump(vae.scaler, os.path.join(scaler_path, "scaler.pkl"))

for epoch in range(EPOCHS):
    # Initialize epoch metrics
    epoch_total_loss = tf.keras.metrics.Mean()
    epoch_reco_loss = tf.keras.metrics.Mean()
    epoch_kl_loss = tf.keras.metrics.Mean()
    epoch_vod_loss = tf.keras.metrics.Mean()

    # KL Annealing: gradually increase beta from 0 to TARGET_BETA over WARMUP_EPOCHS
    # This helps prevent posterior collapse and improves training stability
    if epoch < WARMUP_EPOCHS:
        BETA = TARGET_BETA * (epoch / WARMUP_EPOCHS)
    else:
        BETA = TARGET_BETA

    # Train on all batches for this epoch
    for x_batch in tqdm(train_dataset, desc=f"  Batch", leave=False):
        # Run one training step and get losses
        total_loss, reco_loss, kl_loss, vod_loss = train_step(x_batch, BETA)

        # Update epoch-level metrics
        epoch_total_loss.update_state(total_loss)
        epoch_reco_loss.update_state(reco_loss)
        epoch_kl_loss.update_state(kl_loss)
        epoch_vod_loss.update_state(vod_loss)

    # --- VALIDATION METRICS CALCULATION (WEIGHTS + GEOMETRY) ---

    # Inference (Weights MAE)
    val_pred_w_norm, val_pred_p_norm = vae(val_full_normalized, training=False)

    # Calculate Weights MAE (Normalized)
    val_pred_w_flat = tf.reshape(val_pred_w_norm, [-1, 24])
    val_pred_combined = tf.concat([val_pred_w_flat, val_pred_p_norm], axis=1)
    val_mae = tf.reduce_mean(tf.abs(val_full_normalized - val_pred_combined))

    # Geometric Inference (Geometric MAE)
    # We need to denormalize to generate physical geometry
    pred_w_phys, pred_p_phys = vae.scaler.inverse_transform(
        val_pred_w_norm.numpy(), val_pred_p_norm.numpy()
    )

    # Convert back to Tensor to use in CST Layer
    pred_w_phys_t = tf.convert_to_tensor(pred_w_phys, dtype=tf.float32)
    pred_w_phys_reshaped = tf.reshape(pred_w_phys_t, [-1, 2, 12])

    pred_p_phys_t = tf.convert_to_tensor(pred_p_phys, dtype=tf.float32)

    # Generate predicted coordinates
    pred_geo_coords = vae.decoder.cst_transform(pred_w_phys_reshaped, pred_p_phys_t)
    # Calculate mean absolute difference between coordinates (only Y changes, X is fixed)
    # Shape: (Batch, Points, 2). We are calculating point-to-point average distance.
    val_geo_mae = tf.reduce_mean(tf.abs(true_geo_coords - pred_geo_coords))

    # ---------------------------------------------------------

    # Log epoch results (Updated Print Statement)
    elapsed_time = time.time() - start_time

    if VERBOSE > 0:
        print(
            f"\n[Epoch {epoch+1:3d}/{EPOCHS}] "
            f"Total: {epoch_total_loss.result():.5f} | "
            f"Reco: {epoch_reco_loss.result():.5f} | "
            f"KL: {epoch_kl_loss.result():.5f} | "
            f"W-MAE: {val_mae:.5f} | "
            f"Geo-MAE: {val_geo_mae:.5f} | "
            f"Beta: {BETA:.4f} | "
            f"Time: {elapsed_time:7.1f}s"
        )

    # Log metrics to WandB
    if not DEV:
        wandb.log(
            {
                "beta": BETA,
                "epoch_total_loss": epoch_total_loss.result(),
                "epoch_reconstruction_loss": epoch_reco_loss.result(),
                "epoch_kl_loss": epoch_kl_loss.result(),
                "epoch_vod_loss": epoch_vod_loss.result(),
                "val_mae": val_mae.numpy(),
                "val_geo_mae": val_geo_mae.numpy(),
            }
        )

    # Validation and visualization: run inference on validation set
    val_input_tensor = tf.convert_to_tensor(validation_input)
    # training=False disables stochastic sampling for deterministic reconstruction
    reco_weights_norm, reco_params_norm = vae(val_input_tensor, training=False)

    # Denormalize weights and parameters back to physical range
    real_reco_weights, real_reco_params = vae.scaler.inverse_transform(
        reco_weights_norm.numpy(), reco_params_norm.numpy()
    )

    # Generate airfoil coordinates from denormalized Kulfan parameters
    w_tensor = tf.convert_to_tensor(real_reco_weights, dtype=tf.float32)
    p_tensor = tf.convert_to_tensor(real_reco_params, dtype=tf.float32)
    # CST transform produces (Batch, Points, 2) coordinates
    reco_coords = vae.decoder.cst_transform(w_tensor, p_tensor).numpy()

    # Convert coordinates to Airfoil objects for visualization
    reconstructed_airfoils = []
    for coords in reco_coords:
        reconstructed_airfoils.append(Airfoil(coordinates=coords))

    # Save visualization and model checkpoints periodically
    if (epoch + 1) % CHECKPOINT_EPOCHS == 0:
        print(f"  └─ Saving visualization for epoch {epoch+1}...", end="")

        # Save Weights
        weights_fn = f"vae_weights_epoch_{epoch+1}.weights.h5"
        vae.save_weights(os.path.join(models_path, weights_fn))

        # Generate and save comparison plots of original vs reconstructed airfoils
        plot_original_and_reconstruction(
            validation_airfoils,  # Reference airfoils
            reconstructed_airfoils,  # VAE reconstructions
            text_label=f"Epoch: {epoch+1} / Elapsed Time: {elapsed_time:.2f}s",
            save_path=images_path,
            filename=f"reconstruction_epoch_{epoch+1}.png",
            show=False,
        )
        print(" ✓")
