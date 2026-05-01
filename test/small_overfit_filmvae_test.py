import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from aerosandbox import Airfoil
from tqdm import tqdm

PROJECT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

from src.filmvae.train import (
    CONDITION_COLUMNS,
    TRAIN_DATASET,
    build_airfoil_matrix,
    build_condition_matrix,
    build_model,
    load_dataset,
    resolve_condition_columns,
    set_random_seeds,
)
from src.plotting import plot_original_and_reconstruction
from src.scalers.airfoil_scaler import AirfoilScaler
from src.scalers.condition_scaler import ConditionScaler

SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
LATENT_DIM = 16
NPV = 8
LEARNING_RATE = 1e-3
CLIPNORM = 1.0
WARMUP_EPOCHS = 100
TARGET_BETA = 0.01
SMALL_SIZE = 32
AIRFOILS_TO_PLOT = 9
PLOT_EVERY = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small overfit check for the FiLM-conditioned CST VAE."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--npv", type=int, default=NPV)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--clipnorm", type=float, default=CLIPNORM)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--target-beta", type=float, default=TARGET_BETA)
    parser.add_argument("--small-size", type=int, default=SMALL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--condition-columns",
        nargs=2,
        default=list(CONDITION_COLUMNS),
        metavar=("CL_COLUMN", "ALPHA_COLUMN"),
    )
    parser.add_argument("--airfoils-to-plot", type=int, default=AIRFOILS_TO_PLOT)
    parser.add_argument("--plot-every", type=int, default=PLOT_EVERY)
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Skip writing reconstruction images during training.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for the overfit run.",
    )
    return parser.parse_args()


def maybe_init_wandb(args, run_name, hyperparameters):
    if not args.wandb:
        return None

    import wandb

    wandb.init(
        project="FilmCSTVAE",
        config=hyperparameters,
        name=run_name,
        notes="Small overfit validation for FiLM CST VAE",
    )
    return wandb


def prepare_data(args):
    airfoil_dataset = load_dataset(TRAIN_DATASET, "OVERFIT")
    condition_columns = resolve_condition_columns(
        airfoil_dataset,
        args.condition_columns,
    )

    airfoil_matrix = build_airfoil_matrix(airfoil_dataset)
    condition_matrix = build_condition_matrix(airfoil_dataset, condition_columns)
    raw_weights = airfoil_matrix[:, : 2 * args.npv]
    raw_params = airfoil_matrix[:, 2 * args.npv :]

    scaler = AirfoilScaler()
    scaler.fit(raw_weights, raw_params)

    condition_scaler = ConditionScaler()
    condition_scaler.fit(condition_matrix)

    normalized_geometry = scaler.transform(raw_weights, raw_params)
    normalized_condition = condition_scaler.transform(condition_matrix)

    sample_size = min(args.small_size, len(normalized_geometry))
    sample_geometry = normalized_geometry[:sample_size]
    sample_condition = normalized_condition[:sample_size]
    sample_airfoils_df = airfoil_dataset.iloc[:sample_size].reset_index(drop=True)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (sample_geometry, sample_condition)
    )
    train_dataset = train_dataset.shuffle(sample_size, seed=args.seed).batch(
        args.batch_size
    )

    validation_count = min(args.airfoils_to_plot, sample_size)
    validation_airfoils_df = sample_airfoils_df.iloc[:validation_count].reset_index(
        drop=True
    )
    validation_airfoils = [
        Airfoil(coordinates=af["coordinates"], name=af["airfoil_name"])
        for af in validation_airfoils_df.to_dict(orient="records")
    ]
    validation_geometry = tf.convert_to_tensor(
        sample_geometry[:validation_count],
        dtype=tf.float32,
    )
    validation_condition = tf.convert_to_tensor(
        sample_condition[:validation_count],
        dtype=tf.float32,
    )

    return {
        "condition_columns": condition_columns,
        "scaler": scaler,
        "condition_scaler": condition_scaler,
        "train_dataset": train_dataset,
        "validation_airfoils": validation_airfoils,
        "validation_geometry": validation_geometry,
        "validation_condition": validation_condition,
        "sample_size": sample_size,
        "geometry_range": (float(sample_geometry.min()), float(sample_geometry.max())),
        "condition_range": (
            float(sample_condition.min()),
            float(sample_condition.max()),
        ),
    }


def build_train_step(vae, optimizer, npv):
    @tf.function
    def train_step(geometry, condition, beta):
        with tf.GradientTape() as tape:
            pred_weights, pred_params = vae((geometry, condition), training=True)
            true_weights, true_params = tf.split(geometry, [2 * npv, 2], axis=1)
            pred_weights_flat = tf.reshape(pred_weights, [-1, 2 * npv])

            loss_weights = tf.reduce_mean(
                tf.reduce_sum(tf.square(true_weights - pred_weights_flat), axis=1)
            )
            loss_params = tf.reduce_mean(
                tf.reduce_sum(tf.square(true_params - pred_params), axis=1)
            )
            reco_loss = loss_weights + loss_params

            kl_loss = tf.add_n(vae.losses) if vae.losses else tf.constant(0.0)
            total_loss = reco_loss + (beta * kl_loss)

        grads = tape.gradient(total_loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        return total_loss, reco_loss, kl_loss

    return train_step


def save_reconstruction_plot(
    vae,
    validation_airfoils,
    validation_geometry,
    validation_condition,
    images_path,
    epoch,
    elapsed_time,
):
    reco_weights_norm, reco_params_norm = vae(
        (validation_geometry, validation_condition),
        training=False,
    )
    real_reco_weights, real_reco_params = vae.scaler.inverse_transform(
        reco_weights_norm.numpy(),
        reco_params_norm.numpy(),
    )

    reco_coords = vae.decoder.cst_transform(
        tf.convert_to_tensor(real_reco_weights, dtype=tf.float32),
        tf.convert_to_tensor(real_reco_params, dtype=tf.float32),
    ).numpy()

    reconstructed_airfoils = [Airfoil(coordinates=coords) for coords in reco_coords]
    plot_original_and_reconstruction(
        validation_airfoils,
        reconstructed_airfoils,
        text_label=f"Epoch: {epoch} / Elapsed Time: {elapsed_time:.2f}s",
        save_path=images_path,
        filename=f"reconstruction_epoch_{epoch}.png",
        show=False,
    )


def main():
    args = parse_args()
    set_random_seeds(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    run_name = f"FILMVAE_OVERFIT_{time.strftime('%Y%m%d-%H%M%S')}"
    hyperparameters = {
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "learning_rate": args.learning_rate,
        "target_beta": args.target_beta,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "clipnorm": args.clipnorm,
        "npv": args.npv,
        "small_size": args.small_size,
        "condition_columns": list(args.condition_columns),
    }

    prepared = prepare_data(args)
    scaler = prepared["scaler"]
    train_dataset = prepared["train_dataset"]
    validation_airfoils = prepared["validation_airfoils"]
    validation_geometry = prepared["validation_geometry"]
    validation_condition = prepared["validation_condition"]

    print(f"Using {prepared['sample_size']} samples for the overfit run")
    print(
        "Geometry range: "
        f"[{prepared['geometry_range'][0]:.3f}, {prepared['geometry_range'][1]:.3f}]"
    )
    print(
        "Condition range: "
        f"[{prepared['condition_range'][0]:.3f}, {prepared['condition_range'][1]:.3f}]"
    )

    vae = build_model(
        "filmcstvae",
        scaler,
        npv=args.npv,
        latent_dim=args.latent_dim,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
    )
    train_step = build_train_step(vae, optimizer, args.npv)

    wandb = maybe_init_wandb(args, run_name, hyperparameters)

    output_root = Path("models") / "filmcstvae" / "overfit" / run_name.lower()
    images_path = output_root / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    initial_total_loss = None
    final_total_loss = None
    start_time = time.time()

    print("Starting FilmVAE overfit check...")
    for epoch in range(args.epochs):
        epoch_total_loss = tf.keras.metrics.Mean()
        epoch_reco_loss = tf.keras.metrics.Mean()
        epoch_kl_loss = tf.keras.metrics.Mean()

        beta = (
            args.target_beta * (epoch / args.warmup_epochs)
            if epoch < args.warmup_epochs
            else args.target_beta
        )

        if wandb is not None:
            wandb.log({"beta": beta})

        for geometry_batch, condition_batch in tqdm(
            train_dataset,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            total_loss, reco_loss, kl_loss = train_step(
                geometry_batch,
                condition_batch,
                beta,
            )
            epoch_total_loss.update_state(total_loss)
            epoch_reco_loss.update_state(reco_loss)
            epoch_kl_loss.update_state(kl_loss)

        epoch_total_value = float(epoch_total_loss.result().numpy())
        epoch_reco_value = float(epoch_reco_loss.result().numpy())
        epoch_kl_value = float(epoch_kl_loss.result().numpy())
        elapsed_time = time.time() - start_time

        if initial_total_loss is None:
            initial_total_loss = epoch_total_value
        final_total_loss = epoch_total_value

        print(
            f"Epoch {epoch + 1}/{args.epochs}, "
            f"Time: {elapsed_time:.2f}s, "
            f"Total Loss: {epoch_total_value:.4f}, "
            f"Reco Loss: {epoch_reco_value:.4f}, "
            f"KL Loss: {epoch_kl_value:.4f}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "epoch_total_loss": epoch_total_value,
                    "epoch_reconstruction_loss": epoch_reco_value,
                    "epoch_kl_loss": epoch_kl_value,
                }
            )

        should_plot = (
            not args.disable_plots
            and len(validation_airfoils) > 0
            and ((epoch + 1) % args.plot_every == 0 or epoch == args.epochs - 1)
        )
        if should_plot:
            save_reconstruction_plot(
                vae,
                validation_airfoils,
                validation_geometry,
                validation_condition,
                images_path,
                epoch + 1,
                elapsed_time,
            )

    if wandb is not None:
        wandb.finish()

    if initial_total_loss is None or final_total_loss is None:
        raise RuntimeError("No training steps were executed during the overfit run.")

    improvement_ratio = final_total_loss / initial_total_loss
    print(
        f"Initial total loss: {initial_total_loss:.4f} | "
        f"Final total loss: {final_total_loss:.4f} | "
        f"Ratio: {improvement_ratio:.4f}"
    )

    if args.epochs < 2:
        print("Skipping overfit threshold check because epochs < 2.")
    elif improvement_ratio >= 0.8:
        raise RuntimeError(
            "FilmVAE overfit check did not reduce the loss enough. "
            f"Final/initial ratio: {improvement_ratio:.4f}"
        )

    print(f"Overfit check passed. Outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
