import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

import numpy as np
import pandas as pd
import tensorflow as tf
from aerosandbox import Airfoil
from tqdm import tqdm

from src.filmvae import FiLMCSTVariationalAutoencoder
from src.scalers import AirfoilScaler
from src.scalers import ConditionScaler
from src.plotting import plot_original_and_reconstruction
from src.utils import compute_vod_loss

# ============================================================================
# CONFIGURATION AND RANDOM SEEDS
# ============================================================================

SEED = 42
AIRFOILS_TO_PLOT = 9
CHECKPOINT_EPOCHS = 10
VERBOSE = 1
DEV = False

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

EPOCHS = 500
BATCH_SIZE = 32
LATENT_DIM = 16
NPV = 8
LEARNING_RATE = 1e-3
CLIPNORM = 1.0
WARMUP_EPOCHS = 100
TARGET_BETA = 0.01
SMOOTHNESS_WEIGHT = 0
CONDITION_COLUMNS = ("Cl", "alpha")
CONDITION_DIM = 2

TRAIN_DATASET = "train_filmvae_dataset_8.json"
VALIDATION_DATASET = "val_filmvae_dataset_8.json"

MODEL_REGISTRY = {
    "filmcstvae": FiLMCSTVariationalAutoencoder,
}


def set_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a FiLM-conditioned CST VAE on a geometry + 2D condition dataset (Cl, alpha)."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="filmcstvae",
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--npv", type=int, default=NPV)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--clipnorm", type=float, default=CLIPNORM)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--target-beta", type=float, default=TARGET_BETA)
    parser.add_argument("--smoothness-weight", type=float, default=SMOOTHNESS_WEIGHT)
    parser.add_argument(
        "--condition-columns",
        nargs=2,
        default=list(CONDITION_COLUMNS),
        metavar=("CL_COLUMN", "ALPHA_COLUMN"),
        help="Dataset columns used as FiLM conditions, in the order: Cl alpha.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--airfoils-to-plot", type=int, default=AIRFOILS_TO_PLOT)
    parser.add_argument("--checkpoint-epochs", type=int, default=CHECKPOINT_EPOCHS)
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=VERBOSE)
    parser.add_argument("--dev", action="store_true", default=DEV)
    return parser.parse_args()


def build_hyperparameters(args):
    return {
        "model": args.model,
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "learning_rate": args.learning_rate,
        "target_beta": args.target_beta,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "clipnorm": args.clipnorm,
        "smoothness_weight": args.smoothness_weight,
        "npv": args.npv,
        "seed": args.seed,
        "condition_columns": list(args.condition_columns),
    }


def load_dataset(dataset_name, label):
    dataset_path = PROJECT_PATH / "data" / "processed" / dataset_name
    print("\n" + "=" * 70)
    print(f"LOADING {label} DATASET")
    print("=" * 70)
    airfoil_dataset = pd.read_json(dataset_path)
    print(f"✓ Loaded {len(airfoil_dataset)} {label.lower()} samples")
    airfoil_dataset["coordinates"] = airfoil_dataset["coordinates"].apply(
        lambda coords: np.array(coords)
    )
    return airfoil_dataset


def resolve_condition_columns(airfoil_dataset, requested_columns):
    resolved_columns = []
    available_columns = list(airfoil_dataset.columns)

    for requested_column in requested_columns:
        candidates = []
        for column in [
            requested_column,
            requested_column.lower(),
            requested_column.upper(),
            requested_column.capitalize(),
        ]:
            if column not in candidates:
                candidates.append(column)

        matched_column = next(
            (column for column in candidates if column in airfoil_dataset.columns),
            None,
        )
        if matched_column is None:
            raise ValueError(
                "Condition column not found in dataset. "
                f"Requested '{requested_column}', available columns: {available_columns}"
            )

        resolved_columns.append(matched_column)

    if len(set(resolved_columns)) != len(resolved_columns):
        raise ValueError(f"Condition columns must be distinct, got {resolved_columns}.")

    return resolved_columns


def build_airfoil_matrix(airfoil_dataset):
    airfoil_data = np.concatenate(
        [
            airfoil_dataset["upper_weights"],
            airfoil_dataset["lower_weights"],
            airfoil_dataset["TE_thickness"].values[:, None],
            airfoil_dataset["leading_edge_weight"].values[:, None],
        ],
        axis=0,
    ).to_numpy()

    return np.stack(airfoil_data, axis=0).astype(np.float32)


def build_condition_matrix(airfoil_dataset, condition_columns):
    condition_values = airfoil_dataset[condition_columns].to_numpy(dtype=np.float32)
    if np.isnan(condition_values).any():
        raise ValueError(f"Condition columns {condition_columns} contain NaN values.")

    return condition_values.reshape(-1, len(condition_columns))


def build_model(model_name, scaler, npv, latent_dim):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        scaler,
        npv=npv,
        latent_dim=latent_dim,
    )


def initialize_wandb(dev_mode, hyperparameters, timestring):
    if dev_mode:
        print("\n⚠ Development mode enabled - WandB logging disabled")
        return None

    import wandb

    wandb.init(
        project="FilmCSTVAE",
        config=hyperparameters,
        name=f"{hyperparameters['model'].upper()}_{timestring}",
        notes="FiLM-conditioned CST VAE",
    )
    print("\n✓ WandB initialized for experiment tracking")
    return wandb


def save_scalers(scaler, condition_scaler, condition_columns, scaler_path):
    with open(os.path.join(scaler_path, "scaler.pkl"), "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    metadata = condition_scaler.to_metadata(condition_columns)
    with open(
        os.path.join(scaler_path, "condition_scaler.json"),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    args = parse_args()
    timestring = time.strftime("%Y%m%d-%H%M%S")
    set_random_seeds(args.seed)
    hyperparameters = build_hyperparameters(args)

    # ============================================================================
    # DATASET LOADING AND PREPARATION
    # ============================================================================
    airfoil_dataset = load_dataset(TRAIN_DATASET, "TRAINING")
    condition_columns = resolve_condition_columns(
        airfoil_dataset,
        args.condition_columns,
    )

    airfoil_data = build_airfoil_matrix(airfoil_dataset)
    condition_data = build_condition_matrix(airfoil_dataset, condition_columns)

    raw_weights = airfoil_data[:, :-2]
    raw_params = airfoil_data[:, -2:]
    print(
        f"✓ Data shape: Geometry {airfoil_data.shape} | Condition {condition_data.shape}"
    )

    print("✓ Fitting geometry scaler to data...")
    scaler = AirfoilScaler()
    scaler.fit(raw_weights, raw_params)

    print("✓ Fitting condition scaler to data...")
    condition_scaler = ConditionScaler(condition_dim=len(condition_columns))
    condition_scaler.fit(condition_data)

    print(f"  Weight range: ±{np.max(scaler.w_max):.6f}")
    print(f"  Param range:  ±{np.max(scaler.p_max):.6f}")
    print(
        "  Condition ranges: "
        + ", ".join(
            f"{column}=[{min_value:.6f}, {max_value:.6f}]"
            for column, min_value, max_value in zip(
                condition_columns,
                condition_scaler.min_value,
                condition_scaler.max_value,
            )
        )
    )

    normalized_geometry = scaler.transform(raw_weights, raw_params)
    normalized_condition = condition_scaler.transform(condition_data)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (normalized_geometry, normalized_condition)
    )
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)
    print(f"\n✓ Dataset normalized: {len(normalized_geometry)} samples")
    print(
        f"  Batch size: {args.batch_size} | Total batches: {len(normalized_geometry) // args.batch_size}"
    )
    print(
        f"  Geometry range: [{normalized_geometry.min():.3f}, {normalized_geometry.max():.3f}]"
    )
    print(
        f"  Condition range: [{normalized_condition.min():.3f}, {normalized_condition.max():.3f}]"
    )

    # ============================================================================
    # VALIDATION AIRFOILS PREPARATION
    # ============================================================================
    validation_airfoil_dataset = load_dataset(VALIDATION_DATASET, "VALIDATION")
    validation_condition_columns = resolve_condition_columns(
        validation_airfoil_dataset,
        condition_columns,
    )

    validation_airfoils_sample = validation_airfoil_dataset.iloc[
        : args.airfoils_to_plot
    ].reset_index(drop=True)
    print(
        f"✓ Selected {len(validation_airfoils_sample)} airfoils for validation visualization"
    )
    validation_airfoils = [
        Airfoil(coordinates=af["coordinates"], name=af["airfoil_name"])
        for af in validation_airfoils_sample.to_dict(orient="records")
    ]

    weight_dim = 2 * args.npv

    validation_geometry = build_airfoil_matrix(validation_airfoils_sample)
    validation_condition = build_condition_matrix(
        validation_airfoils_sample,
        validation_condition_columns,
    )
    validation_geometry = scaler.transform(
        validation_geometry[:, :weight_dim],
        validation_geometry[:, weight_dim:],
    )
    validation_condition = condition_scaler.transform(validation_condition)

    # ============================================================================
    # MODEL, OPTIMIZER, AND LOSS INITIALIZATION
    # ============================================================================
    vae = build_model(
        args.model,
        scaler,
        npv=args.npv,
        latent_dim=args.latent_dim,
        condition_dim=len(condition_columns),
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
    )

    # ============================================================================
    # TRAINING STEP FUNCTION
    # ============================================================================
    @tf.function
    def train_step(geometry, condition, beta):
        with tf.GradientTape() as tape:
            pred_weights, pred_params = vae((geometry, condition), training=True)

            true_weights, true_params = tf.split(geometry, [2 * args.npv, 2], axis=1)
            pred_weights_flat = tf.reshape(pred_weights, [-1, 2 * args.npv])

            loss_weights = tf.reduce_mean(
                tf.reduce_sum(tf.square(true_weights - pred_weights_flat), axis=1)
            )
            loss_params = tf.reduce_mean(
                tf.reduce_sum(tf.square(true_params - pred_params), axis=1)
            )
            reco_loss = loss_weights + loss_params

            kl_loss = tf.add_n(vae.losses) if vae.losses else tf.constant(0.0)
            pred_coords_norm = vae.decoder.cst_transform(pred_weights, pred_params)
            y_pred_coords_norm = pred_coords_norm[:, :, 1]
            vod_loss = compute_vod_loss(y_pred_coords_norm)

            total_loss = (
                reco_loss + (beta * kl_loss) + (args.smoothness_weight * vod_loss)
            )

        grads = tape.gradient(total_loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        return total_loss, reco_loss, kl_loss, vod_loss

    wandb = initialize_wandb(args.dev, hyperparameters, timestring)

    models_path = PROJECT_PATH / "models" / args.model / timestring / "weights"
    scaler_path = PROJECT_PATH / "models" / args.model / timestring / "scaler"
    images_path = PROJECT_PATH / "models" / args.model / timestring / "images"
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(scaler_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    print("\n" + "=" * 70)
    print("OUTPUT CONFIGURATION")
    print("=" * 70)
    print(f"✓ Model checkpoints: {models_path}")
    print(f"✓ Visualizations:    {images_path}")

    print("Preparing full validation tensors for metrics...")
    val_full_geometry = build_airfoil_matrix(validation_airfoil_dataset)
    val_full_condition = build_condition_matrix(
        validation_airfoil_dataset,
        validation_condition_columns,
    )

    val_full_normalized_geometry = scaler.transform(
        val_full_geometry[:, :weight_dim],
        val_full_geometry[:, weight_dim:],
    )
    val_full_normalized_condition = condition_scaler.transform(val_full_condition)
    print(
        "✓ Full validation set ready: "
        f"geometry {val_full_normalized_geometry.shape} | "
        f"condition {val_full_normalized_condition.shape}"
    )

    print("Generating 'Ground Truth' geometry for geometric metrics...")
    true_weights_tensor = tf.convert_to_tensor(
        val_full_geometry[:, :weight_dim],
        dtype=tf.float32,
    )
    true_params_tensor = tf.convert_to_tensor(
        val_full_geometry[:, weight_dim:],
        dtype=tf.float32,
    )
    true_weights_reshaped = tf.reshape(true_weights_tensor, [-1, 2, args.npv])
    true_geo_coords = vae.decoder.cst_transform(
        true_weights_reshaped,
        true_params_tensor,
    )
    print(f"✓ Target geometry ready. Shape: {true_geo_coords.shape}")

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Model:             {args.model}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Latent Dim:        {args.latent_dim}")
    print(f"  Learning Rate:     {args.learning_rate}")
    print(f"  Warmup Epochs:     {args.warmup_epochs}")
    print(f"  Target Beta:       {args.target_beta}")
    print(f"  Condition Columns: {condition_columns}")
    print(
        f"  Dev Mode:          {'ON (WandB disabled)' if args.dev else 'OFF (WandB enabled)'}"
    )

    input("\nPress Enter to start training...\n")

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    start_time = time.time()
    save_scalers(vae.scaler, condition_scaler, condition_columns, scaler_path)

    for epoch in range(args.epochs):
        epoch_total_loss = tf.keras.metrics.Mean()
        epoch_reco_loss = tf.keras.metrics.Mean()
        epoch_kl_loss = tf.keras.metrics.Mean()
        epoch_vod_loss = tf.keras.metrics.Mean()

        if epoch < args.warmup_epochs:
            beta = args.target_beta * (epoch / args.warmup_epochs)
        else:
            beta = args.target_beta

        for geometry_batch, condition_batch in tqdm(
            train_dataset,
            desc="  Batch",
            leave=False,
        ):
            total_loss, reco_loss, kl_loss, vod_loss = train_step(
                geometry_batch,
                condition_batch,
                beta,
            )
            epoch_total_loss.update_state(total_loss)
            epoch_reco_loss.update_state(reco_loss)
            epoch_kl_loss.update_state(kl_loss)
            epoch_vod_loss.update_state(vod_loss)

        val_pred_w_norm, val_pred_p_norm = vae(
            (val_full_normalized_geometry, val_full_normalized_condition),
            training=False,
        )
        val_pred_w_flat = tf.reshape(val_pred_w_norm, [-1, weight_dim])
        val_pred_geometry = tf.concat([val_pred_w_flat, val_pred_p_norm], axis=1)
        val_mae = tf.reduce_mean(
            tf.abs(val_full_normalized_geometry - val_pred_geometry)
        )

        pred_w_phys, pred_p_phys = vae.scaler.inverse_transform(
            val_pred_w_norm.numpy(),
            val_pred_p_norm.numpy(),
        )
        pred_w_phys_t = tf.convert_to_tensor(pred_w_phys, dtype=tf.float32)
        pred_p_phys_t = tf.convert_to_tensor(pred_p_phys, dtype=tf.float32)
        pred_geo_coords = vae.decoder.cst_transform(pred_w_phys_t, pred_p_phys_t)
        val_geo_mae = tf.reduce_mean(tf.abs(true_geo_coords - pred_geo_coords))

        elapsed_time = time.time() - start_time

        if args.verbose > 0:
            print(
                f"\n[Epoch {epoch+1:3d}/{args.epochs}] "
                f"Total: {epoch_total_loss.result():.5f} | "
                f"Reco: {epoch_reco_loss.result():.5f} | "
                f"KL: {epoch_kl_loss.result():.5f} | "
                f"W-MAE: {val_mae:.5f} | "
                f"Geo-MAE: {val_geo_mae:.5f} | "
                f"Beta: {beta:.4f} | "
                f"Time: {elapsed_time:7.1f}s"
            )

        if wandb is not None:
            wandb.log(
                {
                    "beta": beta,
                    "epoch_total_loss": epoch_total_loss.result(),
                    "epoch_reconstruction_loss": epoch_reco_loss.result(),
                    "epoch_kl_loss": epoch_kl_loss.result(),
                    "epoch_vod_loss": epoch_vod_loss.result(),
                    "val_mae": val_mae.numpy(),
                    "val_geo_mae": val_geo_mae.numpy(),
                }
            )

        reco_weights_norm, reco_params_norm = vae(
            (validation_geometry, validation_condition),
            training=False,
        )

        real_reco_weights, real_reco_params = vae.scaler.inverse_transform(
            reco_weights_norm.numpy(),
            reco_params_norm.numpy(),
        )

        w_tensor = tf.convert_to_tensor(real_reco_weights, dtype=tf.float32)
        p_tensor = tf.convert_to_tensor(real_reco_params, dtype=tf.float32)
        reco_coords = vae.decoder.cst_transform(w_tensor, p_tensor).numpy()

        reconstructed_airfoils = []
        for coords in reco_coords:
            reconstructed_airfoils.append(Airfoil(coordinates=coords))

        if (epoch + 1) % args.checkpoint_epochs == 0:
            print(f"  └─ Saving visualization for epoch {epoch+1}...", end="")
            weights_fn = f"vae_weights_epoch_{epoch+1}.weights.h5"
            vae.save_weights(os.path.join(models_path, weights_fn))
            plot_original_and_reconstruction(
                validation_airfoils,
                reconstructed_airfoils,
                text_label=f"Epoch: {epoch+1} / Elapsed Time: {elapsed_time:.2f}s",
                save_path=images_path,
                filename=f"reconstruction_epoch_{epoch+1}.png",
                show=False,
            )
            print(" ✓")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
