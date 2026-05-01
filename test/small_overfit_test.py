import os
import sys
import random
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
from src.scalers.airfoil_scaler import AirfoilScaler

from src.airfoil import airfoil_modifications

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Hyperparameters ---
EPOCHS = 1000  # 5000

BATCH_SIZE = 32
LATENT_DIM = 16
NPV = (
    12  # Number of CST coefficients MUST BE EQUAL TO THE ONE USED IN DATASET GENERATION
)
LEARNING_RATE = 1e-3
CLIPNORM = 1.0  # Gradient clipping norm value
WARMUP_EPOCHS = 1000  # Number of epochs for KL annealing warm-up
TARGET_BETA = 0.005  # Weight for KL Divergence Loss

SMALL_SIZE = 32  # Number of samples to overfit on

HYPERPARAMETERS = {
    "epochs": EPOCHS,
    "latent_dim": LATENT_DIM,
    "initial_learning_rate": LEARNING_RATE,
    "target_beta": TARGET_BETA,
    "warmup_epochs": WARMUP_EPOCHS,
    "batch_size": BATCH_SIZE,
    "clipnorm": CLIPNORM,
}

proj_path = "./"

# --- Dataset Loading ---
dataset_path = Path(proj_path) / "data" / "processed" / "kulfan_dataset_75.json"

print("Loading dataset...")
airfoil_dataset = pd.read_json(dataset_path)

# Fix coordinates
airfoil_dataset["coordinates"] = airfoil_dataset["coordinates"].apply(
    lambda coords: np.array(coords)
)

# Prepare the data: concatenate lower and upper weights along with TE thickness and leading edge weight
airfoil_data = (
    airfoil_dataset["kulfan_parameters"]
    .apply(
        lambda p: np.concatenate(
            [
                p["lower_weights"],
                p["upper_weights"],
                [p["TE_thickness"]],
                [p["leading_edge_weight"]],
            ],
            axis=0,
        )
    )
    .to_numpy()
)

airfoil_data = np.stack(airfoil_data, axis=0).astype(np.float32)

raw_weights = airfoil_data[:, :-2]
raw_params = airfoil_data[:, -2:]

print(f"Original Dataset Size: {len(raw_params)}")

# Filter 1: Reasonable Thickness (e.g., < 5cm)
# If your data is normalized, ensure this check happens on raw physical values
valid_te = raw_params[:, 0] < 0.05

# Filter 2: Reasonable Weights (e.g., between -2 and 2)
# Any weight larger than this is likely a glitch
valid_weights = np.all((raw_weights > -2.0) & (raw_weights < 2.0), axis=1)

# Combine and Apply
valid_mask = valid_te & valid_weights
raw_weights = raw_weights[valid_mask]
raw_params = raw_params[valid_mask]

print(f"Cleaned Dataset Size: {len(raw_params)}")

# Now fit the scaler on clean data
scaler = AirfoilScaler()
scaler.fit(raw_weights, raw_params)
print(f"Max Weight Value: {np.max(scaler.w_max)}")
print(f"Max Param Value: {np.max(scaler.p_max)}")

normalized_data = scaler.transform(raw_weights, raw_params)

small_normalized_data = normalized_data[:SMALL_SIZE]

# Create a tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(small_normalized_data)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
print(f"Dataset loaded and normalized: {len(small_normalized_data)} samples")
print(
    f"Data Range Check -> Min: {small_normalized_data.min():.2f}, Max: {small_normalized_data.max():.2f}"
)

# Picking airfoils for validation and plotting
validation_airfoils_df = airfoil_dataset.iloc[:9].reset_index(drop=True)
validation_airfoils = [
    Airfoil(coordinates=af["coordinates"], name=af["airfoil_name"])
    for af in validation_airfoils_df.to_dict(orient="records")
]

validation_input = (
    validation_airfoils_df["kulfan_parameters"]
    .apply(
        lambda p: np.concatenate(
            [
                p["lower_weights"],
                p["upper_weights"],
                [p["TE_thickness"]],
                [p["leading_edge_weight"]],
            ],
            axis=0,
        )
    )
    .to_list()
)

validation_input = tf.convert_to_tensor(validation_input, dtype=tf.float32)
weights = validation_input[:, :24]
params = validation_input[:, 24:]
validation_input = scaler.transform(weights, params)

# --- Instantiate Model, Optimizer, and Loss ---
vae = CSTVariationalAutoencoder(scaler, npv=NPV, latent_dim=LATENT_DIM)

best_loss = float("inf")
wait = 0
current_lr = LEARNING_RATE  # Seu LR inicial
optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, clipnorm=CLIPNORM)

# Main Loss (Reconstruction Loss)
# reconstruction_loss = tf.keras.losses.MeanSquaredError()
reconstruction_loss = tf.keras.losses.MeanAbsoluteError()


# --- Training Step (for one batch) ---
@tf.function
def train_step(data, beta):
    """
    Runs one training step with robust Sum-Squared Error loss.
    """
    with tf.GradientTape() as tape:
        # 1. Forward Pass
        # training=True enables the SamplingLayer noise
        reconstruction = vae(data, training=True)
        _, pred_weights, pred_params = reconstruction

        # 2. Split Targets
        # data shape: (Batch, 26)
        true_weights, true_params = tf.split(data, [2 * NPV, 2], axis=1)

        # 3. Flatten Predictions for Loss Calculation
        # pred_weights shape is (Batch, 2, 12) -> Flatten to (Batch, 24)
        pred_weights_flat = tf.reshape(pred_weights, [-1, 2 * NPV])

        # 4. Calculate Reconstruction Loss (Sum of Squares)
        # We use SUM to match the magnitude of the physics
        loss_weights = tf.reduce_mean(
            tf.reduce_sum(tf.square(true_weights - pred_weights_flat), axis=1)
        )
        loss_params = tf.reduce_mean(
            tf.reduce_sum(tf.square(true_params - pred_params), axis=1)
        )

        reco_loss = loss_weights + loss_params

        # 5. KL Loss (Already calculated inside the model via self.add_loss)
        kl_loss = sum(vae.losses)

        # 6. Total Loss
        total_loss = reco_loss + (beta * kl_loss)

    # 7. Backpropagation
    grads = tape.gradient(total_loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    return total_loss, reco_loss, kl_loss


wandb.init(
    project="CSTVAE",
    config=HYPERPARAMETERS,
    name=f"VAE_{time.strftime('%Y%m%d-%H%M%S')}",
    notes="Dense Arch + Linear Output + Sum Loss + Scaler",
)

VERBOSE = 1

# Ensure the models directory exists
models_path = Path(proj_path) / "models" / "cstvae" / time.strftime("%Y%m%d-%H%M%S")
images_path = Path(proj_path) / "images" / "cstvae" / time.strftime("%Y%m%d-%H%M%S")
os.makedirs(models_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# --- The Main Training Loop ---
print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):

    # Reset metrics at the start of each epoch
    epoch_total_loss = tf.keras.metrics.Mean()
    epoch_reco_loss = tf.keras.metrics.Mean()
    epoch_kl_loss = tf.keras.metrics.Mean()

    # Linear ramp for BETA
    if epoch < WARMUP_EPOCHS:
        BETA = TARGET_BETA * (epoch / WARMUP_EPOCHS)
    else:
        BETA = TARGET_BETA

    wandb.log({"beta": BETA})

    # Iterate over each batch in the dataset
    for x_batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        # Run one training step
        total_loss, reco_loss, kl_loss = train_step(x_batch, BETA)

        # Update the epoch's average loss
        epoch_total_loss.update_state(total_loss)
        epoch_reco_loss.update_state(reco_loss)
        epoch_kl_loss.update_state(kl_loss)

    # --- End of Epoch ---
    elapsed_time = time.time() - start_time

    if VERBOSE > 0:
        print(
            f"Epoch {epoch+1}/{EPOCHS}, "
            f"Time: {elapsed_time:.2f}s, "
            f"Total Loss: {epoch_total_loss.result():.4f}, "
            f"Reco Loss: {epoch_reco_loss.result():.4f}, "
            f"KL Loss: {epoch_kl_loss.result():.4f}"
        )

    wandb.log(
        {
            "epoch_total_loss": epoch_total_loss.result(),
            "epoch_reconstruction_loss": epoch_reco_loss.result(),
            "epoch_kl_loss": epoch_kl_loss.result(),
        }
    )

    # print("Generating and plotting airfoils...")

    # 1. Get Normalized Predictions
    # CRITICAL: training=False disables noise for smooth plots
    # Ensure input is a Tensor (Fixes the ValueError)
    val_input_tensor = tf.convert_to_tensor(validation_input)
    _, reco_weights_norm, reco_params_norm = vae(val_input_tensor, training=False)

    # 2. Denormalize to Physical Values
    real_reco_weights, real_reco_params = vae.scaler.inverse_transform(
        reco_weights_norm.numpy(), reco_params_norm.numpy()
    )

    # 3. Generate Coordinates using the Model's Internal Logic
    w_tensor = tf.convert_to_tensor(real_reco_weights, dtype=tf.float32)
    p_tensor = tf.convert_to_tensor(real_reco_params, dtype=tf.float32)

    # This generates (Batch, Points, 2)
    reco_coords = vae.decoder.cst_transform(w_tensor, p_tensor).numpy()

    # 4. Wrap in Airfoil objects for your plotting function
    reconstructed_airfoils = []
    for coords in reco_coords:
        reconstructed_airfoils.append(Airfoil(coordinates=coords))

    # Save model checkpoints every 5 epochs
    if (epoch + 1) % 10 == 0:
        # print(f"--- Saving model checkpoint for epoch {epoch+1} ---")
        # vae.save_weights(f"{models_path}/model_epoch_{epoch+1}.weights.h5")

        plot_original_and_reconstruction(
            validation_airfoils,
            reconstructed_airfoils,
            text_label=f"Epoch: {epoch+1} / Elapsed Time: {elapsed_time:.2f}s",
            save_path=images_path,
            filename=f"reconstruction_epoch_{epoch+1}.png",
            show=False,
        )
