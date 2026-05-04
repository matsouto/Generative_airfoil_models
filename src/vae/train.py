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

from src.plotting import plot_original_and_reconstruction
from src.scalers import AirfoilScaler
from src.vae import ConvCSTVariationalAutoencoder
from src.vae import CSTVariationalAutoencoder

# ============================================================================
# CONFIGURATION AND RANDOM SEEDS
# ============================================================================

SEED = 42
AIRFOILS_TO_PLOT = 16
CHECKPOINT_EPOCHS = 10
VERBOSE = 1
DEV = False

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

EPOCHS = 150
BATCH_SIZE = 32
LATENT_DIM = 16
NPV = 8
LEARNING_RATE = 1e-3
CLIPNORM = 1.0
WEIGHT_DECAY = 0.0
WARMUP_EPOCHS = 100
TARGET_BETA = 0.01
BETA_ANNEALING = "cyclical"
ACTIVE_UNITS_THRESHOLD = 1e-2

TRAIN_DATASET = "train_kulfan_dataset_8.json"
VALIDATION_DATASET = "val_kulfan_dataset_8.json"

MODEL_REGISTRY = {
    "cstvae": CSTVariationalAutoencoder,
    "convcstvae": ConvCSTVariationalAutoencoder,
}


def set_random_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VAE models on the Kulfan parameter dataset."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="cstvae",
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--npv", type=int, default=NPV)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--clipnorm", type=float, default=CLIPNORM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--target-beta", type=float, default=TARGET_BETA)
    parser.add_argument(
        "--beta-annealing",
        choices=["cyclical", "linear", "constant"],
        default=BETA_ANNEALING,
        help=(
            "Beta annealing strategy. 'cyclical' repeats a warmup cycle, "
            "'linear' performs a single warmup, and 'constant' keeps target beta fixed."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--airfoils-to-plot", type=int, default=AIRFOILS_TO_PLOT)
    parser.add_argument("--checkpoint-epochs", type=int, default=CHECKPOINT_EPOCHS)
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=VERBOSE)
    parser.add_argument("--dev", action="store_true", default=DEV)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Disable saving checkpoints, scalers, and reconstruction images for sweep runs.",
    )
    return parser.parse_args()


def build_hyperparameters(args):
    return {
        "model": args.model,
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "target_beta": args.target_beta,
        "beta_annealing": args.beta_annealing,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "clipnorm": args.clipnorm,
        "npv": args.npv,
        "seed": args.seed,
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


def build_airfoil_matrix(airfoil_dataset):
    airfoil_data = (
        airfoil_dataset["kulfan_parameters"]
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
        .to_numpy()
    )

    return np.stack(airfoil_data, axis=0).astype(np.float32)


def sample_validation_airfoils(airfoil_dataset, sample_size, random_state=None):
    if sample_size > len(airfoil_dataset):
        raise ValueError(
            "Validation sample size cannot exceed dataset size. "
            f"Requested {sample_size}, available {len(airfoil_dataset)}."
        )

    return airfoil_dataset.sample(
        n=sample_size,
        replace=False,
        random_state=random_state,
    ).reset_index(drop=True)


def compute_beta(epoch, warmup_epochs, target_beta, annealing_type):
    if target_beta <= 0:
        return 0.0

    if annealing_type == "constant":
        return target_beta

    if warmup_epochs <= 0:
        return target_beta

    if annealing_type == "linear":
        progress = min(epoch, warmup_epochs) / warmup_epochs
        return target_beta * progress

    if annealing_type == "cyclical":
        cycle_epoch = epoch % warmup_epochs
        ramp_epochs = max(1, int(np.ceil(warmup_epochs * 0.5)))
        progress = min(cycle_epoch, ramp_epochs) / ramp_epochs
        return target_beta * progress

    raise ValueError(f"Unsupported beta annealing type: {annealing_type}")


def build_model(model_name, scaler, npv, latent_dim):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(scaler, npv=npv, latent_dim=latent_dim)


def compute_active_units(z_mean, threshold=ACTIVE_UNITS_THRESHOLD):
    latent_variances = tf.math.reduce_variance(z_mean, axis=0)
    active_units = tf.reduce_sum(
        tf.cast(latent_variances > threshold, tf.float32),
        axis=0,
    )
    return active_units, latent_variances


def initialize_wandb(dev_mode, hyperparameters, timestring):
    if dev_mode:
        print("\n⚠ Development mode enabled - WandB logging disabled")
        return None

    import wandb

    wandb.init(
        project="CSTVAE",
        config=hyperparameters,
        name=f"{hyperparameters['model'].upper()}_{timestring}",
        notes="Dense Arch + Linear Output + Sum Loss + Scaler",
    )
    print("\n✓ WandB initialized for experiment tracking")
    return wandb


def apply_wandb_overrides(args, wandb_module):
    if wandb_module is None:
        return args

    config = wandb_module.config
    override_fields = {
        "epochs",
        "batch_size",
        "latent_dim",
        "npv",
        "learning_rate",
        "clipnorm",
        "weight_decay",
        "warmup_epochs",
        "target_beta",
        "beta_annealing",
        "seed",
        "airfoils_to_plot",
        "checkpoint_epochs",
        "verbose",
    }

    for field_name in override_fields:
        if field_name in config:
            setattr(args, field_name, config[field_name])

    return args


def save_scaler(scaler, scaler_path):
    with open(os.path.join(scaler_path, "scaler.pkl"), "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    metadata = {
        "w_max": np.asarray(scaler.w_max, dtype=float).tolist(),
        "p_max": np.asarray(scaler.p_max, dtype=float).tolist(),
    }
    with open(
        os.path.join(scaler_path, "scaler_metadata.json"),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    args = parse_args()
    timestring = time.strftime("%Y%m%d-%H%M%S")
    hyperparameters = build_hyperparameters(args)
    wandb = initialize_wandb(args.dev, hyperparameters, timestring)
    args = apply_wandb_overrides(args, wandb)
    if wandb is not None:
        wandb.config.update(build_hyperparameters(args), allow_val_change=True)
    set_random_seeds(args.seed)

    # ============================================================================
    # DATASET LOADING AND PREPARATION
    # ============================================================================
    airfoil_dataset = load_dataset(TRAIN_DATASET, "TRAINING")
    airfoil_data = build_airfoil_matrix(airfoil_dataset)

    raw_weights = airfoil_data[:, :-2]
    raw_params = airfoil_data[:, -2:]
    print(f"✓ Data shape: Weights {raw_weights.shape} | Params {raw_params.shape}")

    print("✓ Fitting scaler to data...")
    scaler = AirfoilScaler()
    scaler.fit(raw_weights, raw_params)

    print(f"  Weight range: ±{np.max(scaler.w_max):.6f}")
    print(f"  Param range:  ±{np.max(scaler.p_max):.6f}")

    normalized_data = scaler.transform(raw_weights, raw_params)

    train_dataset = tf.data.Dataset.from_tensor_slices(normalized_data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)
    print(f"\n✓ Dataset normalized: {len(normalized_data)} samples")
    print(
        f"  Batch size: {args.batch_size} | Total batches: {len(normalized_data) // args.batch_size}"
    )
    print(f"  Data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

    # ============================================================================
    # VALIDATION AIRFOILS PREPARATION
    # ============================================================================
    validation_airfoil_dataset = load_dataset(VALIDATION_DATASET, "VALIDATION")

    validation_airfoils_sample = sample_validation_airfoils(
        validation_airfoil_dataset,
        args.airfoils_to_plot,
        random_state=args.seed,
    )
    print(
        f"✓ Selected {len(validation_airfoils_sample)} fixed airfoils for validation visualization"
    )
    validation_airfoils = [
        Airfoil(coordinates=af["coordinates"], name=af["airfoil_name"])
        for af in validation_airfoils_sample.to_dict(orient="records")
    ]

    validation_input = (
        validation_airfoils_sample["kulfan_parameters"]
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

    validation_input = tf.convert_to_tensor(validation_input, dtype=tf.float32)
    weight_dim = 2 * args.npv
    weights = validation_input[:, :weight_dim]
    params = validation_input[:, weight_dim:]
    validation_input = scaler.transform(weights, params)

    # ============================================================================
    # MODEL, OPTIMIZER, AND LOSS INITIALIZATION
    # ============================================================================
    vae = build_model(args.model, scaler, npv=args.npv, latent_dim=args.latent_dim)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
    )
    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "clipnorm": args.clipnorm,
    }
    if args.weight_decay > 0:
        optimizer = tf.keras.optimizers.AdamW(
            weight_decay=args.weight_decay,
            **optimizer_kwargs,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)

    # ============================================================================
    # TRAINING STEP FUNCTION
    # ============================================================================
    @tf.function
    def train_step(data, beta):
        with tf.GradientTape() as tape:
            reconstruction = vae(data, training=True)
            pred_weights, pred_params = reconstruction

            true_weights, true_params = tf.split(data, [2 * args.npv, 2], axis=1)
            pred_weights_flat = tf.reshape(pred_weights, [-1, 2 * args.npv])

            loss_weights = tf.reduce_mean(
                tf.reduce_sum(tf.square(true_weights - pred_weights_flat), axis=1),
                axis=0,
            )
            loss_params = tf.reduce_mean(
                tf.reduce_sum(tf.square(true_params - pred_params), axis=1),
                axis=0,
            )
            reco_loss = loss_weights + loss_params

            kl_loss = tf.add_n(vae.losses) if vae.losses else tf.constant(0.0)
            total_loss = reco_loss + (beta * kl_loss)

        grads = tape.gradient(total_loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        return total_loss, reco_loss, kl_loss

    models_path = PROJECT_PATH / "models" / args.model / timestring / "weights"
    scaler_path = PROJECT_PATH / "models" / args.model / timestring / "scaler"
    images_path = PROJECT_PATH / "models" / args.model / timestring / "images"
    if not args.sweep:
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(scaler_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
    print("\n" + "=" * 70)
    print("OUTPUT CONFIGURATION")
    print("=" * 70)
    if args.sweep:
        print("✓ Sweep mode: output artifacts disabled")
    else:
        print(f"✓ Model checkpoints: {models_path}")
        print(f"✓ Visualizations:    {images_path}")

    print("Preparing full validation tensor for metrics...")
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

    val_full_tensor = tf.convert_to_tensor(val_full_vectors, dtype=tf.float32)
    val_full_normalized = scaler.transform(
        val_full_tensor[:, :weight_dim],
        val_full_tensor[:, weight_dim:],
    )
    print(f"✓ Full validation set ready: {val_full_normalized.shape}")

    print("Generating 'Ground Truth' geometry for geometric metrics...")
    true_weights_tensor = val_full_tensor[:, :weight_dim]
    true_params_tensor = val_full_tensor[:, weight_dim:]
    true_weights_reshaped = tf.reshape(true_weights_tensor, [-1, 2, args.npv])
    true_geo_coords = vae.decoder.cst_transform(
        true_weights_reshaped,
        true_params_tensor,
    )
    print(f"✓ Target geometry ready. Shape: {true_geo_coords.shape}")

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Latent Dim:     {args.latent_dim}")
    print(f"  Learning Rate:  {args.learning_rate}")
    print(f"  Warmup Epochs:  {args.warmup_epochs}")
    print(f"  Target Beta:    {args.target_beta}")
    print(f"  Beta Annealing: {args.beta_annealing}")
    print(
        f"  Dev Mode:       {'ON (WandB disabled)' if args.dev else 'OFF (WandB enabled)'}"
    )

    if sys.stdin is not None and sys.stdin.isatty():
        try:
            input("\nPress Enter to start training...\n")
        except EOFError:
            print(
                "\nNon-interactive stdin detected. Starting training automatically..."
            )
    else:
        print("\nNon-interactive run detected. Starting training automatically...")

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    start_time = time.time()
    if not args.sweep:
        save_scaler(vae.scaler, scaler_path)

    for epoch in range(args.epochs):
        epoch_total_loss = tf.keras.metrics.Mean()
        epoch_reco_loss = tf.keras.metrics.Mean()
        epoch_kl_loss = tf.keras.metrics.Mean()

        beta = compute_beta(
            epoch,
            args.warmup_epochs,
            args.target_beta,
            args.beta_annealing,
        )

        for x_batch in tqdm(train_dataset, desc="  Batch", leave=False):
            total_loss, reco_loss, kl_loss = train_step(x_batch, beta)
            epoch_total_loss.update_state(total_loss)
            epoch_reco_loss.update_state(reco_loss)
            epoch_kl_loss.update_state(kl_loss)

        val_pred_w_norm, val_pred_p_norm = vae(val_full_normalized, training=False)
        val_z_mean, _ = vae.encoder(val_full_normalized)
        active_units, latent_variances = compute_active_units(val_z_mean)
        val_pred_w_flat = tf.reshape(val_pred_w_norm, [-1, weight_dim])
        val_pred_combined = tf.concat([val_pred_w_flat, val_pred_p_norm], axis=1)
        val_mae = tf.reduce_mean(
            tf.abs(val_full_normalized - val_pred_combined),
            axis=None,
        )

        pred_w_phys, pred_p_phys = vae.scaler.inverse_transform(
            val_pred_w_norm.numpy(),
            val_pred_p_norm.numpy(),
        )
        pred_w_phys_t = tf.convert_to_tensor(pred_w_phys, dtype=tf.float32)
        pred_p_phys_t = tf.convert_to_tensor(pred_p_phys, dtype=tf.float32)
        pred_geo_coords = vae.decoder.cst_transform(pred_w_phys_t, pred_p_phys_t)
        val_geo_mae = tf.reduce_mean(
            tf.abs(true_geo_coords - pred_geo_coords),
            axis=None,
        )

        elapsed_time = time.time() - start_time

        if args.verbose > 0:
            print(
                f"\n[Epoch {epoch+1:3d}/{args.epochs}] "
                f"Total: {epoch_total_loss.result():.5f} | "
                f"Reco: {epoch_reco_loss.result():.5f} | "
                f"KL: {epoch_kl_loss.result():.5f} | "
                f"W-MAE: {val_mae:.5f} | "
                f"Geo-MAE: {val_geo_mae:.5f} | "
                f"Active Units: {active_units:.0f} | "
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
                    "val_mae": val_mae.numpy(),
                    "val_geo_mae": val_geo_mae.numpy(),
                    "active_units": active_units.numpy(),
                    "latent_variance_mean": tf.reduce_mean(
                        latent_variances,
                        axis=0,
                    ).numpy(),
                }
            )

        if not args.sweep and (epoch + 1) % args.checkpoint_epochs == 0:
            reco_weights_norm, reco_params_norm = vae(validation_input, training=False)

            real_reco_weights, real_reco_params = vae.scaler.inverse_transform(
                reco_weights_norm.numpy(),
                reco_params_norm.numpy(),
            )

            w_tensor = tf.convert_to_tensor(real_reco_weights, dtype=tf.float32)
            p_tensor = tf.convert_to_tensor(real_reco_params, dtype=tf.float32)
            reco_coords = vae.decoder.cst_transform(w_tensor, p_tensor).numpy()

            reconstructed_airfoils = [
                Airfoil(coordinates=coords) for coords in reco_coords
            ]

            print(f"  └─ Saving visualization for epoch {epoch+1}...", end="")
            weights_fn = f"vae_weights_epoch_{epoch+1}.weights.h5"
            vae.save_weights(os.path.join(models_path, weights_fn))
            plot_original_and_reconstruction(
                validation_airfoils,
                reconstructed_airfoils,
                text_label=(
                    f"Epoch: {epoch+1} / Total Loss: {epoch_total_loss.result():.5f} "
                    f"/ Elapsed Time: {elapsed_time:.2f}s"
                ),
                save_path=images_path,
                filename=f"reconstruction_epoch_{epoch+1}.png",
                show=False,
            )
            print(" ✓")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
