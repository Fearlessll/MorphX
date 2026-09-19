"""Unified MFM/MSFM training entry point for HCC and LUAD.

Examples:

    python -m prognosis.train_mfm_msfm \
        --cohort hcc --dataset tcga --data-root /data/HCC_path \
        --feature-families all --model macro

    python -m prognosis.train_mfm_msfm \
        --cohort luad --dataset tcga --data-root /data/LUNG_path \
        --feature-families all --model fusion \
        --micro-feature-template /data/micro/fold_{fold}.csv
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from configs.cohorts import CohortSpec, default_dataset_dir, default_wsi_list_path, get_cohort, parse_feature_families
from prognosis.clinical import load_survival_labels
from prognosis.data_loaders import FusionFeatureMapDataset, MacroFeatureMapDataset
from prognosis.Networks.fusion_net import FusionNet
from prognosis.Networks.resnet import resnet10
from prognosis.utils import cindex_lifeline, count_parameters, cox_log_rank, cox_loss, modified_cox_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _load_split(split_dir: Path, fold: int) -> Tuple[List[str], List[str]]:
    split_path = split_dir / f"split_data_fold_{fold}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Fixed split file does not exist: {split_path}")
    with split_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    train = payload.get("train_data")
    validation = payload.get("test_data", payload.get("val_data"))
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError(f"Split file must contain train_data and test_data lists: {split_path}")
    if not train or not validation or len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError(f"Split has empty or duplicate entries: {split_path}")
    if set(train) & set(validation):
        raise ValueError(f"Training and validation overlap in {split_path}")
    return train, validation


def _resolve_feature_paths(names: Iterable[str], feature_map_dir: Path) -> List[str]:
    paths = []
    missing = []
    for name in names:
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = feature_map_dir / candidate
        if not candidate.is_file():
            missing.append(str(candidate))
        paths.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} feature maps are missing under {feature_map_dir}; examples: {missing[:5]}"
        )
    return paths


def _default_feature_map_dir(
    data_root: Path, cohort: CohortSpec, dataset: str, image_size: int
) -> Path:
    dataset_dir = default_dataset_dir(data_root, cohort, dataset)
    base = (
        dataset_dir
        / "processed_data"
        / "feature_maps"
        / "concat_feature_maps"
        / f"{cohort.total_channels}d"
    )
    candidates = [
        base / "initial" / "final_feature_maps" / str(image_size),
        base / "initial_tumour" / "final_feature_maps_new" / str(image_size),
    ]
    present = [candidate for candidate in candidates if candidate.is_dir()]
    if len(present) == 1:
        return present[0]
    raise FileNotFoundError(
        f"Expected one processed {cohort.total_channels}-channel {image_size}x{image_size} "
        f"feature-map directory under {base}; found {present}. Set --feature-map-dir explicitly."
    )


def _format_template(template: Optional[str], fold: int) -> Optional[Path]:
    if not template:
        return None
    return Path(template.format(fold=fold))


def _build_model(args, input_channels: int, macro_checkpoint: Optional[Path] = None):
    spec = get_cohort(args.cohort)
    if args.model == "macro":
        return resnet10(
            first_covd_param=args.first_conv,
            input_channel_num=input_channels,
            output_use_sigmoid=getattr(args, "train_output", "sigmoid") == "sigmoid",
            backbone_width=spec.backbone_width,
        )
    model = FusionNet(
        macro_first_covd_param=args.first_conv,
        macro_input_channel_num=input_channels,
        micro_feature_dim=args.micro_feature_dim,
        macro_feature_dim=spec.backbone_width * 8,
        backbone_width=spec.backbone_width,
        macro_best_ckpt_path=str(macro_checkpoint) if macro_checkpoint else None,
        output_use_sigmoid=True,
    )
    if getattr(args, "freeze_macro", False):
        for parameter in model.macro_net.parameters():
            parameter.requires_grad = False
    return model


def _make_dataset(
    args,
    paths: Sequence[str],
    events: np.ndarray,
    times: np.ndarray,
    channel_indices: Sequence[int],
):
    if args.model == "macro":
        return MacroFeatureMapDataset(
            paths,
            events,
            times,
            channel_indices=channel_indices,
            image_size=args.image_size,
        )
    micro_csv = (args.micro_features_csv if args.stage == "final"
                 else _format_template(args.micro_feature_template, args.current_fold))
    if micro_csv is None:
        raise ValueError("--micro-feature-template is required for fusion training.")
    return FusionFeatureMapDataset(
        paths,
        events,
        times,
        micro_features_csv=micro_csv,
        micro_feature_dim=args.micro_feature_dim,
        channel_indices=channel_indices,
        image_size=args.image_size,
    )


def _predict(model, loader, device, model_type: str):
    model.eval()
    # The historical training loss consumes sigmoid output; risk analyses use
    # the pre-sigmoid score from the same head.
    previous = model.output_use_sigmoid
    model.output_use_sigmoid = False
    risks, times, events = [], [], []
    try:
        with torch.no_grad():
            for inputs, batch_times, batch_events in loader:
                if model_type == "macro":
                    scores = model(inputs.to(device).float())[1]
                else:
                    micro, macro = inputs
                    scores = model(micro.to(device).float(), macro.to(device).float())
                risks.append(scores.reshape(-1).cpu().numpy())
                times.append(batch_times.numpy())
                events.append(batch_events.numpy())
    finally:
        model.output_use_sigmoid = previous
    return (
        np.concatenate(risks),
        np.concatenate(times),
        np.concatenate(events),
    )


def _evaluate(model, loader, device, model_type: str) -> Dict[str, float]:
    risk, times, events = _predict(model, loader, device, model_type)
    return {
        "cindex": cindex_lifeline(risk, events, times),
        "logrank_p": cox_log_rank(risk, events, times),
        "n": int(len(risk)),
        "events": int(events.sum()),
    }


def _save_predictions(path: Path, names, risk, times, events) -> None:
    import pandas as pd

    pd.DataFrame(
        {"wsi_name": [Path(name).name for name in names], "risk_score": risk,
         "survival_months": times, "event": events.astype(int)}
    ).to_csv(path, index=False)


def train_fold(args, fold: int, device: torch.device, spec: CohortSpec, channel_indices):
    args.current_fold = fold
    train_names, validation_names = _load_split(args.splits_dir, fold)
    train_paths = _resolve_feature_paths(train_names, args.feature_map_dir)
    validation_paths = _resolve_feature_paths(validation_names, args.feature_map_dir)

    train_events, train_times = load_survival_labels(
        train_names, spec, args.dataset, data_root=args.data_root, clinical_csv=args.clinical_csv
    )
    validation_events, validation_times = load_survival_labels(
        validation_names, spec, args.dataset, data_root=args.data_root, clinical_csv=args.clinical_csv
    )
    train_set = _make_dataset(args, train_paths, train_events, train_times, channel_indices)
    validation_set = _make_dataset(
        args, validation_paths, validation_events, validation_times, channel_indices
    )
    event_counts = np.bincount(train_events.astype(int), minlength=2)
    sampling_weights = [1.0 / event_counts[int(value)] for value in train_events]
    sampler = WeightedRandomSampler(sampling_weights, len(train_set), replacement=False)
    train_loader = DataLoader(
        train_set,
        batch_size=len(train_set) if args.batch_size <= 0 else args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=len(validation_set) if args.batch_size <= 0 else args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    macro_checkpoint = _format_template(args.macro_checkpoint_template, fold)
    if macro_checkpoint and not macro_checkpoint.is_file():
        raise FileNotFoundError(f"Macro checkpoint does not exist: {macro_checkpoint}")
    model = _build_model(args, len(channel_indices), macro_checkpoint).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, threshold=0.01, patience=1
    )
    output_dir = args.output_dir / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_cindex = -float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, batch_times, batch_events in train_loader:
            batch_times = batch_times.to(device)
            batch_events = batch_events.to(device)
            if args.model == "macro":
                scores = model(inputs.to(device).float())[1]
            else:
                micro, macro = inputs
                scores = model(micro.to(device).float(), macro.to(device).float())
            loss_function = modified_cox_loss if args.loss_function == "source_modified" else cox_loss
            loss = loss_function(batch_times, batch_events, scores)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        scheduler.step(float(loss.detach().cpu()))
        validation_metrics = _evaluate(model, validation_loader, device, args.model)
        train_metrics = _evaluate(model, train_loader, device, args.model)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        score = validation_metrics["cindex"]
        score_for_comparison = score if np.isfinite(score) else -float("inf")
        checkpoint = {
            "fold": fold,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "metrics": record,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if epoch == 0 or score_for_comparison > best_cindex:
            best_cindex = score_for_comparison
            torch.save(checkpoint, output_dir / "best.pt")
            risk, times, events = _predict(model, validation_loader, device, args.model)
            _save_predictions(output_dir / "validation_predictions.csv", validation_names, risk, times, events)
        print(
            f"fold={fold} epoch={epoch + 1}/{args.epochs} "
            f"loss={record['train_loss']:.5f} "
            f"val_cindex={validation_metrics['cindex']:.4f}"
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"loss_function": args.loss_function, "history": history,
                   "best_cindex": best_cindex}, handle, indent=2, default=_json_default)
    return best_cindex


def _median_optimal_epochs(cv_output_dir: Path, loss_function: str) -> int:
    """Use discovery-fold validation results from the same loss protocol."""

    best_epochs = []
    for fold in range(10):
        path = cv_output_dir / f"fold_{fold}" / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("loss_function") != loss_function:
            raise ValueError(
                f"Fold {fold} does not record --loss-function {loss_function}: {path}. "
                "Rerun all ten discovery folds with the selected loss."
            )
        history = metrics["history"]
        scores = np.asarray([row["validation"]["cindex"] for row in history], dtype=float)
        if not np.isfinite(scores).any():
            raise ValueError(f"No finite validation C-index for fold {fold}")
        best_epochs.append(int(np.nanargmax(scores)) + 1)
    return int(np.ceil(np.median(best_epochs)))


def train_final(args, device: torch.device, spec: CohortSpec, channel_indices):
    """Fit one model on every discovery WSI, without consulting external data."""

    list_path = default_wsi_list_path(args.data_root, spec, args.dataset)
    with list_path.open("r", encoding="utf-8-sig") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError(f"Malformed full-discovery WSI list: {list_path}")
    paths = _resolve_feature_paths(names, args.feature_map_dir)
    events, times = load_survival_labels(
        names, spec, args.dataset, data_root=args.data_root, clinical_csv=args.clinical_csv
    )
    dataset = _make_dataset(args, paths, events, times, channel_indices)
    counts = np.bincount(events.astype(int), minlength=2)
    weights = [1.0 / counts[int(value)] for value in events]
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=False)
    train_loader = DataLoader(dataset, batch_size=len(dataset) if args.batch_size <= 0 else args.batch_size,
                              sampler=sampler, num_workers=args.workers, pin_memory=device.type == "cuda")
    predict_loader = DataLoader(dataset, batch_size=len(dataset) if args.batch_size <= 0 else args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    macro_checkpoint = args.macro_checkpoint if args.model == "fusion" else None
    model = _build_model(args, len(channel_indices), macro_checkpoint).to(device)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, threshold=0.01, patience=1
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, batch_times, batch_events in train_loader:
            if args.model == "macro":
                scores = model(inputs.to(device).float())[1]
            else:
                micro, macro = inputs
                scores = model(micro.to(device).float(), macro.to(device).float())
            loss_function = modified_cox_loss if args.loss_function == "source_modified" else cox_loss
            loss = loss_function(batch_times.to(device), batch_events.to(device), scores)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        mean_loss = total_loss / len(train_loader)
        scheduler.step(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": mean_loss})
        print(f"final epoch={epoch + 1}/{args.epochs} loss={mean_loss:.5f}")
    checkpoint = {"stage": "final", "epoch": args.epochs - 1, "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(), "config": vars(args), "history": history}
    torch.save(checkpoint, args.output_dir / "final.pt")
    risk, times, events = _predict(model, predict_loader, device, args.model)
    _save_predictions(args.output_dir / "discovery_predictions.csv", names, risk, times, events)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"n": len(names), "epochs": args.epochs, "history": history}, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MorphX MFM or MSFM survival models.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--dataset", choices=("tcga", "kmmufh", "gd"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clinical-csv", type=Path, default=None)
    parser.add_argument("--feature-map-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--feature-families", default="all")
    parser.add_argument("--model", choices=("macro", "fusion"), default="macro")
    parser.add_argument("--stage", choices=("cv", "final"), default="cv")
    parser.add_argument("--micro-feature-template", default=None)
    parser.add_argument("--micro-features-csv", type=Path, default=None)
    parser.add_argument("--micro-feature-dim", type=int, default=None)
    parser.add_argument("--macro-checkpoint-template", default=None)
    parser.add_argument("--macro-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-macro", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--first-conv", nargs=3, type=int, default=(3, 2, 1))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=None, help="Default follows the cohort and model protocol.")
    parser.add_argument("--cv-output-dir", type=Path, default=None,
                        help="For final stage, derive epochs from ten discovery CV folds.")
    parser.add_argument("--batch-size", type=int, default=10, help="0 uses one full batch per fold.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--loss-function", choices=("cox", "source_modified"), default="cox",
                        help="Default: standard Cox partial likelihood with Breslow ties and "
                             "per-event mean reduction; source_modified reproduces the old two-term loss.")
    parser.add_argument("--train-output", choices=("sigmoid", "logit"), default="sigmoid",
                        help="Training score activation; logit enables the MFM no-sigmoid ablation.")
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/morphx"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_cohort(args.cohort)
    if args.dataset not in spec.clinical_files:
        raise ValueError(f"{args.dataset} is not a valid dataset for {args.cohort}.")
    spec.validate_training_dataset(args.dataset)
    if args.model == "fusion" and args.micro_feature_dim is None:
        raise ValueError("--micro-feature-dim is required for fusion training.")
    if args.model == "fusion" and args.train_output != "sigmoid":
        raise ValueError("--train-output logit is defined only for the MFM ablation")
    if args.model == "fusion" and args.micro_feature_dim != spec.total_channels:
        raise ValueError(f"MSFM requires {spec.total_channels} micro features for {spec.name}.")
    if args.stage == "cv" and args.model == "fusion" and (
        not args.micro_feature_template or not args.macro_checkpoint_template
    ):
        raise ValueError("Fusion CV requires --micro-feature-template and --macro-checkpoint-template.")
    if args.stage == "final" and args.model == "fusion" and (
        args.micro_features_csv is None or args.macro_checkpoint is None
    ):
        raise ValueError("Final fusion requires --micro-features-csv and --macro-checkpoint.")
    if args.stage == "final" and args.epochs is None and args.cv_output_dir is None:
        raise ValueError("Final training requires --cv-output-dir or an explicit --epochs.")
    if args.stage == "final" and args.epochs is None:
        args.epochs = _median_optimal_epochs(args.cv_output_dir, args.loss_function)
    if args.epochs is None:
        args.epochs = spec.macro_epochs if args.model == "macro" else spec.fusion_epochs
    if args.learning_rate is None:
        args.learning_rate = spec.macro_learning_rate if args.model == "macro" else spec.fusion_learning_rate
    channel_indices, family_names = parse_feature_families(args.feature_families, spec)
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.splits_dir = (args.splits_dir or default_dataset_dir(args.data_root, spec, args.dataset)).resolve()
    args.feature_map_dir = (
        args.feature_map_dir
        or _default_feature_map_dir(
            args.data_root, spec, args.dataset, args.image_size
        )
    ).resolve()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.clinical_csv is None:
        args.clinical_csv = None
    else:
        args.clinical_csv = args.clinical_csv.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "selected_feature_families": family_names,
                "selected_channel_indices": channel_indices,
            },
            handle,
            indent=2,
            default=_json_default,
        )
    _seed_everything(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Feature families: {family_names}; channels: {len(channel_indices)}")
    print(f"Trainable parameters are reported per fold.")
    if args.stage == "final":
        train_final(args, device, spec, channel_indices)
    else:
        results = []
        for fold in args.folds:
            _seed_everything(args.seed)
            result = train_fold(args, fold, device, spec, channel_indices)
            results.append({"fold": fold, "best_cindex": result})
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
