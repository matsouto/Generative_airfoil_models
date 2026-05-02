import argparse
import json
import random
import sys
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from tqdm.auto import tqdm

PROJECT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

from src.airfoil.compute_airfoil_quality import QualityError, compute_airfoil_quality

DEFAULT_SEED = 42
DEFAULT_N_WEIGHTS_PER_SIDE = 8
DEFAULT_N_POINTS_PER_SIDE = 75
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_AUGMENTATION_COPIES = 2

LW_CAP = 8.0
UW_CAP = 8.0
THICKNESS_CAP = 10.0
LE_CAP = 10.0

MAX_CAMBER_DELTA = 0.010
MAX_THICKNESS_SCALE_DELTA = 0.080
MAX_LOCAL_BUMP = 0.006

DEFAULT_SIMULATION_CONFIG = {
    "alpha_start": -6.0,
    "alpha_stop": 16.0,
    "alpha_step": 0.50,
    "Re": 1e6,
    "mach": 0.0,
    "n_crit": 9.0,
    "xtr_upper": 1.0,
    "xtr_lower": 1.0,
    "model_size": "xlarge",
    "include_360_deg_effects": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FiLMVAE train/validation datasets from the AeroSandbox airfoil database."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-weights-per-side",
        type=int,
        default=DEFAULT_N_WEIGHTS_PER_SIDE,
    )
    parser.add_argument(
        "--n-points-per-side",
        type=int,
        default=DEFAULT_N_POINTS_PER_SIDE,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--enable-augmentation",
        action="store_true",
        help="Generate smooth augmented copies for each approved airfoil.",
    )
    parser.add_argument(
        "--augmentation-copies-per-airfoil",
        type=int,
        default=DEFAULT_AUGMENTATION_COPIES,
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_PATH / "data" / "processed",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Optional explicit path for the train dataset JSON.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Optional explicit path for the validation dataset JSON.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional explicit path for the metadata JSON.",
    )
    parser.add_argument(
        "--alpha-start", type=float, default=DEFAULT_SIMULATION_CONFIG["alpha_start"]
    )
    parser.add_argument(
        "--alpha-stop", type=float, default=DEFAULT_SIMULATION_CONFIG["alpha_stop"]
    )
    parser.add_argument(
        "--alpha-step", type=float, default=DEFAULT_SIMULATION_CONFIG["alpha_step"]
    )
    parser.add_argument("--re", type=float, default=DEFAULT_SIMULATION_CONFIG["Re"])
    parser.add_argument("--mach", type=float, default=DEFAULT_SIMULATION_CONFIG["mach"])
    parser.add_argument(
        "--n-crit", type=float, default=DEFAULT_SIMULATION_CONFIG["n_crit"]
    )
    parser.add_argument(
        "--xtr-upper", type=float, default=DEFAULT_SIMULATION_CONFIG["xtr_upper"]
    )
    parser.add_argument(
        "--xtr-lower", type=float, default=DEFAULT_SIMULATION_CONFIG["xtr_lower"]
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_SIMULATION_CONFIG["model_size"],
    )
    parser.add_argument(
        "--include-360-deg-effects",
        action="store_true",
        default=DEFAULT_SIMULATION_CONFIG["include_360_deg_effects"],
    )
    return parser.parse_args()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_default_simulation_config(args):
    return {
        "alpha_start": args.alpha_start,
        "alpha_stop": args.alpha_stop,
        "alpha_step": args.alpha_step,
        "Re": args.re,
        "mach": args.mach,
        "n_crit": args.n_crit,
        "xtr_upper": args.xtr_upper,
        "xtr_lower": args.xtr_lower,
        "model_size": args.model_size,
        "include_360_deg_effects": args.include_360_deg_effects,
    }


def load_quality_filtered_airfoils():
    airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
    raw_airfoil_database = [
        asb.Airfoil(name=filename.stem).normalize()
        for filename in airfoil_database_path.glob("*.dat")
    ]

    quality_airfoil_database = []
    failed_quality = []
    for airfoil in tqdm(raw_airfoil_database, desc="Quality check"):
        try:
            compute_airfoil_quality(airfoil, airfoil_database_path)
            quality_airfoil_database.append(airfoil)
        except QualityError as exc:
            failed_quality.append((airfoil.name, str(exc)))

    failed_quality_df = pd.DataFrame(
        failed_quality,
        columns=["airfoil_name", "reason"],
    )
    return quality_airfoil_database, failed_quality_df


def build_airfoil_records(
    airfoil_records,
    n_weights_per_side,
    n_points_per_side,
):
    rows = []
    rejected = []

    for record in tqdm(airfoil_records, desc="Building airfoil records"):
        airfoil = record["airfoil"]
        try:
            standardized_airfoil = airfoil.normalize().repanel(n_points_per_side)
            parameters = get_kulfan_parameters(
                coordinates=standardized_airfoil.coordinates,
                n_weights_per_side=n_weights_per_side,
                normalize_coordinates=True,
                use_leading_edge_modification=True,
            )
        except Exception as exc:
            rejected.append((airfoil.name, f"kulfan_failed: {exc}"))
            continue

        lower_weights = np.asarray(parameters["lower_weights"], dtype=float)
        upper_weights = np.asarray(parameters["upper_weights"], dtype=float)
        te_thickness = float(parameters["TE_thickness"])
        leading_edge_weight = float(parameters["leading_edge_weight"])

        if (
            np.any(lower_weights > LW_CAP)
            or np.any(upper_weights > UW_CAP)
            or te_thickness > THICKNESS_CAP
            or leading_edge_weight < -LE_CAP
            or leading_edge_weight > LE_CAP
        ):
            rejected.append((airfoil.name, "outlier_filtered"))
            continue

        coords = standardized_airfoil.coordinates
        thickness_proxy = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))

        row = {
            "airfoil_name": airfoil.name,
            "coordinates": coords.tolist(),
            "lower_weights": lower_weights.tolist(),
            "upper_weights": upper_weights.tolist(),
            "TE_thickness": te_thickness,
            "leading_edge_weight": leading_edge_weight,
            "shape": list(coords.shape),
            "points": int(coords.shape[0]),
            "thickness_proxy": thickness_proxy,
        }
        row.update({key: value for key, value in record.items() if key != "airfoil"})
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(
        rejected, columns=["airfoil_name", "reason"]
    )


def distort_coordinates(coords, rng, n_points_per_side):
    coords = np.asarray(coords, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1].copy()

    envelope = np.sin(np.pi * x) ** 2
    sinusoid = np.sin(
        2 * np.pi * rng.uniform(0.5, 1.5) * x + rng.uniform(0.0, 2 * np.pi)
    )
    camber_shift = rng.uniform(-MAX_CAMBER_DELTA, MAX_CAMBER_DELTA)
    thickness_scale = 1.0 + rng.uniform(
        -MAX_THICKNESS_SCALE_DELTA,
        MAX_THICKNESS_SCALE_DELTA,
    )
    local_bump = rng.uniform(-MAX_LOCAL_BUMP, MAX_LOCAL_BUMP) * envelope * sinusoid

    y = y + (camber_shift * envelope) + local_bump
    y = y * (1.0 + (thickness_scale - 1.0) * envelope)

    augmented = (
        asb.Airfoil(
            name="augmented",
            coordinates=np.column_stack([x, y]),
        )
        .normalize()
        .repanel(n_points_per_side)
    )
    return augmented.coordinates.tolist()


def maybe_augment_airfoils(base_airfoils_dataset, args):
    augmented_airfoils_dataset = pd.DataFrame()
    augmentation_rejections = pd.DataFrame(columns=["airfoil_name", "reason"])

    if not args.enable_augmentation:
        return (
            base_airfoils_dataset.copy(),
            augmented_airfoils_dataset,
            augmentation_rejections,
        )

    rng = np.random.default_rng(args.augmentation_seed)
    augmented_records = []
    iterator = tqdm(
        base_airfoils_dataset.iterrows(),
        total=len(base_airfoils_dataset),
        desc="Augmenting airfoils",
    )
    for _, row in iterator:
        for copy_idx in range(args.augmentation_copies_per_airfoil):
            coords = distort_coordinates(
                row["coordinates"],
                rng,
                args.n_points_per_side,
            )
            augmented_records.append(
                {
                    "airfoil": asb.Airfoil(
                        name=f"{row['airfoil_name']}_aug_{copy_idx:02d}",
                        coordinates=np.asarray(coords, dtype=float),
                    ),
                    "source_airfoil_name": row["source_airfoil_name"],
                    "augmentation_tag": f"aug_{copy_idx:02d}",
                    "is_augmented": True,
                }
            )

    augmented_airfoils_dataset, augmentation_rejections = build_airfoil_records(
        augmented_records,
        n_weights_per_side=args.n_weights_per_side,
        n_points_per_side=args.n_points_per_side,
    )

    if len(augmented_airfoils_dataset) > 0:
        airfoils_dataset = pd.concat(
            [base_airfoils_dataset, augmented_airfoils_dataset],
            ignore_index=True,
        )
    else:
        airfoils_dataset = base_airfoils_dataset.copy()

    return (
        airfoils_dataset.reset_index(drop=True),
        augmented_airfoils_dataset,
        augmentation_rejections,
    )


def build_simulation_config(_, default_simulation_config):
    return dict(default_simulation_config)


def split_airfoils_grouped_by_source(airfoils_dataset, validation_fraction, seed):
    unique_groups = sorted(airfoils_dataset["source_airfoil_name"].unique())
    rng = np.random.default_rng(seed)
    n_val_groups = max(1, int(round(len(unique_groups) * validation_fraction)))
    val_groups = set(
        rng.choice(unique_groups, size=n_val_groups, replace=False).tolist()
    )

    train_airfoils = airfoils_dataset[
        ~airfoils_dataset["source_airfoil_name"].isin(val_groups)
    ].reset_index(drop=True)
    val_airfoils = airfoils_dataset[
        airfoils_dataset["source_airfoil_name"].isin(val_groups)
    ].reset_index(drop=True)
    return train_airfoils, val_airfoils, unique_groups, val_groups


def make_alpha_grid(config):
    return np.arange(
        config["alpha_start"],
        config["alpha_stop"] + (0.5 * config["alpha_step"]),
        config["alpha_step"],
        dtype=float,
    )


def run_neuralfoil_for_airfoil(row):
    config = dict(row["simulation_config"])
    alpha_values = make_alpha_grid(config)

    airfoil = asb.Airfoil(
        name=row["airfoil_name"],
        coordinates=np.asarray(row["coordinates"], dtype=float),
    )

    repeated = lambda value: np.full_like(alpha_values, value, dtype=float)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_values,
        Re=repeated(config["Re"]),
        mach=repeated(config["mach"]),
        n_crit=repeated(config["n_crit"]),
        xtr_upper=repeated(config["xtr_upper"]),
        xtr_lower=repeated(config["xtr_lower"]),
        model_size=config["model_size"],
        include_360_deg_effects=config["include_360_deg_effects"],
    )

    cl = np.asarray(aero["CL"], dtype=float)
    cd = np.asarray(aero["CD"], dtype=float)
    cm = np.asarray(aero["CM"], dtype=float)

    valid_mask = (
        np.isfinite(alpha_values) & np.isfinite(cl) & np.isfinite(cd) & np.isfinite(cm)
    )
    polar_rows = []
    for idx in np.nonzero(valid_mask)[0]:
        polar_rows.append(
            {
                "airfoil_id": row["airfoil_id"],
                "profile_id": row["airfoil_id"],
                "airfoil_name": row["airfoil_name"],
                "source_airfoil_name": row["source_airfoil_name"],
                "augmentation_tag": row["augmentation_tag"],
                "is_augmented": bool(row["is_augmented"]),
                "coordinates": row["coordinates"],
                "lower_weights": row["lower_weights"],
                "upper_weights": row["upper_weights"],
                "TE_thickness": float(row["TE_thickness"]),
                "leading_edge_weight": float(row["leading_edge_weight"]),
                "shape": row["shape"],
                "points": int(row["points"]),
                "alpha": float(alpha_values[idx]),
                "Cl": float(cl[idx]),
                "Cd": float(cd[idx]),
                "Cm": float(cm[idx]),
                "Re": float(config["Re"]),
                "mach": float(config["mach"]),
                "n_crit": float(config["n_crit"]),
                "xtr_upper": float(config["xtr_upper"]),
                "xtr_lower": float(config["xtr_lower"]),
                "model_size": config["model_size"],
            }
        )

    return polar_rows


def expand_airfoils_with_neuralfoil(airfoils_df, split_name):
    expanded_rows = []
    failed_airfoils = []

    iterator = tqdm(
        airfoils_df.iterrows(),
        total=len(airfoils_df),
        desc=f"NeuralFoil {split_name}",
    )
    for _, row in iterator:
        try:
            expanded_rows.extend(run_neuralfoil_for_airfoil(row))
        except Exception as exc:
            failed_airfoils.append(
                (row["airfoil_name"], f"{type(exc).__name__}: {exc}")
            )

    expanded_df = pd.DataFrame(expanded_rows)
    failed_df = pd.DataFrame(failed_airfoils, columns=["airfoil_name", "reason"])
    return expanded_df, failed_df


def resolve_output_paths(args):
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_output_path = args.train_output or processed_dir / (
        f"train_filmvae_dataset_{args.n_weights_per_side}.json"
    )
    val_output_path = args.val_output or processed_dir / (
        f"val_filmvae_dataset_{args.n_weights_per_side}.json"
    )
    metadata_output_path = args.metadata_output or processed_dir / (
        f"filmvae_dataset_metadata_{args.n_weights_per_side}.json"
    )
    return train_output_path, val_output_path, metadata_output_path


def print_summary_table(label, dataframe):
    print(f"{label}: {len(dataframe)}")


def main():
    args = parse_args()
    set_random_seeds(args.seed)
    default_simulation_config = build_default_simulation_config(args)

    approved_airfoils, failed_quality = load_quality_filtered_airfoils()
    print_summary_table(
        "Raw approved airfoils",
        pd.DataFrame({"airfoil_name": [af.name for af in approved_airfoils]}),
    )
    print_summary_table("Rejected on quality", failed_quality)

    original_records = [
        {
            "airfoil": airfoil,
            "source_airfoil_name": airfoil.name,
            "augmentation_tag": "original",
            "is_augmented": False,
        }
        for airfoil in approved_airfoils
    ]

    base_airfoils_dataset, build_rejections = build_airfoil_records(
        original_records,
        n_weights_per_side=args.n_weights_per_side,
        n_points_per_side=args.n_points_per_side,
    )
    print_summary_table("Base airfoils", base_airfoils_dataset)
    print_summary_table("Rejected on Kulfan build", build_rejections)

    airfoils_dataset, augmented_airfoils_dataset, augmentation_rejections = (
        maybe_augment_airfoils(
            base_airfoils_dataset,
            args,
        )
    )
    print_summary_table("Augmented airfoils", augmented_airfoils_dataset)
    print_summary_table("Rejected on augmentation build", augmentation_rejections)

    airfoils_dataset = airfoils_dataset.reset_index(drop=True)
    airfoils_dataset["airfoil_id"] = airfoils_dataset.index.map(
        lambda idx: f"airfoil_{idx:06d}"
    )
    airfoils_dataset["simulation_config"] = airfoils_dataset.apply(
        build_simulation_config,
        axis=1,
        default_simulation_config=default_simulation_config,
    )

    train_airfoils, val_airfoils, unique_groups, val_groups = (
        split_airfoils_grouped_by_source(
            airfoils_dataset,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )
    print(f"Unique source profiles: {len(unique_groups)}")
    print(f"Validation source profiles: {len(val_groups)}")
    print(f"Train grouped profiles: {len(train_airfoils)}")
    print(f"Val grouped profiles:   {len(val_airfoils)}")

    train_filmvae_dataset, train_nf_failures = expand_airfoils_with_neuralfoil(
        train_airfoils,
        "train",
    )
    val_filmvae_dataset, val_nf_failures = expand_airfoils_with_neuralfoil(
        val_airfoils,
        "validation",
    )

    print(f"Train FiLMVAE samples: {len(train_filmvae_dataset)}")
    print(f"Val FiLMVAE samples:   {len(val_filmvae_dataset)}")
    print(f"Train NeuralFoil failures: {len(train_nf_failures)}")
    print(f"Val NeuralFoil failures:   {len(val_nf_failures)}")

    train_output_path, val_output_path, metadata_output_path = resolve_output_paths(
        args
    )
    train_filmvae_dataset.to_json(train_output_path)
    val_filmvae_dataset.to_json(val_output_path)

    metadata = {
        "n_weights_per_side": args.n_weights_per_side,
        "n_points_per_side": args.n_points_per_side,
        "train_samples": int(len(train_filmvae_dataset)),
        "val_samples": int(len(val_filmvae_dataset)),
        "augmentation_enabled": bool(args.enable_augmentation),
        "augmentation_copies_per_airfoil": int(args.augmentation_copies_per_airfoil),
        "default_simulation_config": default_simulation_config,
        "condition_columns": ["Cl", "alpha"],
        "validation_fraction": float(args.validation_fraction),
        "seed": int(args.seed),
        "quality_rejections": int(len(failed_quality)),
        "kulfan_rejections": int(len(build_rejections)),
        "augmentation_rejections": int(len(augmentation_rejections)),
        "train_neuralfoil_failures": int(len(train_nf_failures)),
        "val_neuralfoil_failures": int(len(val_nf_failures)),
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved train dataset: {train_output_path}")
    print(f"Saved val dataset:   {val_output_path}")
    print(f"Saved metadata:      {metadata_output_path}")


if __name__ == "__main__":
    main()
