# MorphX research code

MorphX is one Python project for the HCC and LUAD survival models in the manuscript. It trains the Morphological Features Model (MFM), uses integrated gradients to select key tiles, trains the Multi-Scale Features Model (MSFM) from supplied micro-feature tables, evaluates fixed models externally, and exports risks for endpoint and biomarker analyses.

This repository contains the MorphX model and analysis code plus channel metadata. It contains no patient data, WSI, masks, trained weights, experiment outputs, Pathomics package, or code that extracts features from WSI or nuclei. Supply the prepared feature maps and micro-feature CSVs described below.

| Cohort | Discovery | Locked external | Tissue | Texture | Nuclear/cell | Full map | MFM encoder |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HCC | TCGA-LIHC | KMMUFH | 8 | 40 | 120 | 168 | 1024 |
| LUAD | TCGA-LUAD | GDPH (`gd`) | 5 | 40 | 120 | 165 | 2048 |

`configs/cohorts.py` contains the cohort-specific file names, channels, model widths, and schedules. HCC and LUAD use the same main implementation.

## Install and run the demo

Use Python 3.10 or later and a compatible PyTorch build.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python demo/run_synthetic_demo.py
```

The CPU demo exercises MFM, attribution, top-K selection, MSFM, fixed external evaluation, and a final discovery MFM fit using random maps and **mock micro features**. Its metrics have no biological or manuscript meaning. See [demo/README.md](demo/README.md).

## Data contract and read-only audit

Each dataset needs a JSON WSI list, a clinical CSV, and prepared `256 × 256 × C` HWC NumPy maps. Channels are ordered tissue, texture, nuclear/cell as specified by `configs/metadata/feats_HCC.cvs` and `configs/metadata/feats_LUNG.cvs`. TCGA discovery also needs the original `split_data_fold_0.json` through `split_data_fold_9.json`. Training never regenerates these splits. GDPH's manuscript analysis list is `GD/all_data_mil.json` (301 slides); `GD/all_data.json` is a different 308-slide list.

```bash
python -m prognosis.preflight --cohort hcc --dataset tcga --data-root /path/to/HCC_path
python -m prognosis.preflight --cohort hcc --dataset kmmufh --data-root /path/to/HCC_path
python -m prognosis.preflight --cohort luad --dataset tcga --data-root /path/to/LUNG_path --feature-map-dir /path/to/TCGA/processed_165_channel_maps/256
python -m prognosis.preflight --cohort luad --dataset gd --data-root /path/to/LUNG_path --feature-map-dir /path/to/GD/processed_165_channel_maps/256
```

`--data-root` locates clinical and split files; `--feature-map-dir` may point to another disk. The audit checks every listed map's shape and clinical match plus all ten fixed splits. It reads metadata and never changes clinical data. Actual audit findings are in [docs/reproducibility.md](docs/reproducibility.md).

## Prepared inputs and map assembly

Feature extraction from WSI, tissue segmentation, nucleus masks, and selected tiles is outside this repository. Supply either final model maps directly or precomputed tissue (8 HCC / 5 LUAD channels), texture (40), and nuclear/cell (120) maps on the same source grid. The model-side preparation commands can assemble and normalize those existing arrays:

```bash
python -m data_preprocess.concat_feature_maps --cohort hcc --wsi-list /path/to/HCC_path/TCGA/all_data.json --tissue-dir /path/to/tissue_maps --texture-dir /path/to/texture_maps --nuclear-dir /path/to/nuclear_maps --output-dir /path/to/full_maps
python -m data_preprocess.normalize_feature_maps --wsi-list /path/to/HCC_path/TCGA/all_data.json --input-dir /path/to/full_maps --mask-dir /path/to/tissue_masks --method initial --output-dir /path/to/processed_maps
```

Use `--cohort luad` for 165 channels. The normalizer writes `processed_maps/geometry/<slide>.json` for mapping attribution back to the source grid. If you supply final maps made elsewhere, provide matching geometry JSONs before running `prognosis.key_tiles`; historical maps without a verified inverse transform cannot yield reliable WSI tile coordinates. For `zscore` or `minmax`, fit `--stats-path` on discovery data with `--fit-stats` and apply it unchanged externally. These commands assemble existing features; they do not extract them from images.

MSFM additionally requires one CSV row per WSI with `wsi_id` and numeric columns `feature_0` through `feature_167` (HCC) or `feature_164` (LUAD). Supply separate CSVs for each MFM fold and dataset, produced from that fold's selected tiles; the data loader checks names, dimensions, duplicates, and nonfinite values.

## Discovery ten-fold MFM and MSFM

Example HCC training; LUAD uses `--cohort luad` and an explicit map directory when clinical and map roots differ:

```bash
python -m prognosis.train_mfm_msfm --cohort hcc --dataset tcga --data-root /path/to/HCC_path --feature-map-dir /path/to/processed_maps --model macro --feature-families all --output-dir runs/hcc_mfm
```

Except for the loss, defaults follow the source scripts: HCC MFM 15 epochs at 6.6e-4 and MSFM 10 at 6.6e-5; LUAD MFM 40 at 6.6e-4 and MSFM 20 at 9.9e-5. Training uses batch size 10, Adam, event-weighted sampling without replacement, and the source learning-rate scheduler. The loss consumes a sigmoid score; predictions export the pre-sigmoid logit. Only discovery-fold validation C-index selects `best.pt`. Each fold also writes `last.pt`, `metrics.json`, and validation predictions.

For this project the author selected **standard Cox partial likelihood**. `--loss-function cox` is the default for MFM and MSFM, in both cross-validation and final training. It uses Breslow handling for tied event times and averages the negative log likelihood over observed events. The historical source scripts instead use a two-term `modified_cox_loss`, available explicitly as `--loss-function source_modified` for source-code comparisons. Retrain the ten discovery folds, their dependent MSFM runs, and final models under Cox before attributing any manuscript numbers to this protocol. Final training refuses cross-validation metrics recorded under a different or unknown loss. The source also instantiated an unused SelfNorm layer; its checkpoint keys remain so historical MFM checkpoints load strictly. See [docs/reproducibility.md](docs/reproducibility.md).

For the manuscript's MFM no-sigmoid ablation, add `--train-output logit` to a separate MFM CV run. The default `sigmoid` follows the source training path. The inference CSV always contains the unbounded head logit. Other feature-family and top-K ablations use `--feature-families` and `--top-k` in separate runs; keep the original discovery folds and locked external cohort for each run.

For each MFM fold, use that fold's fixed checkpoint to attribute TCGA and external maps separately. Defaults are 50-step zero-baseline integrated gradients, sum of absolute attribution across channels, and top 16 tiles. Repeat for folds 0 through 9 and each dataset:

```bash
python -m prognosis.key_tiles --cohort hcc --checkpoint runs/hcc_mfm/fold_0/best.pt --wsi-list /path/to/HCC_path/TCGA/all_data.json --feature-map-dir /path/to/processed_maps --geometry-dir /path/to/processed_maps/geometry --output-csv runs/hcc_tiles/fold_0.csv --feature-scores-csv runs/hcc_tiles/fold_0_feature_scores.csv
```

The selector excludes padding and the empty tissue class, then records each position on the source feature grid. Its output is a tile manifest for an external feature-extraction workflow. That workflow must return fold-specific micro-feature CSVs in the format above; no tile feature extractor is bundled here.

Train MSFM from the supplied fold-specific micro features and MFM checkpoints. Its macro encoder is initialized from MFM and fine-tuned unless `--freeze-macro` is set:

```bash
python -m prognosis.train_mfm_msfm --cohort hcc --dataset tcga --data-root /path/to/HCC_path --feature-map-dir /path/to/processed_maps --model fusion --micro-feature-dim 168 --micro-feature-template 'runs/hcc_micro/fold_{fold}.csv' --macro-checkpoint-template 'runs/hcc_mfm/fold_{fold}/best.pt' --output-dir runs/hcc_msfm
```

For LUAD use `--micro-feature-dim 165`. Feature-family ablations always slice absolute indices from full 168/165-channel maps. They do not reinterpret reduced 40- or 120-channel maps as full maps.

The LUAD reviewer-response normal-region ablation is part of the main preprocessing package. Candidate A retains available tissue outside all existing tissue classes; candidate B retains available tissue outside the `te` tumor class. It writes masked 165-channel maps and a `mask_qc.csv` with retained pixels and scaling factors. Train MFM against each output directory using the same fixed TCGA folds, then evaluate each fixed checkpoint on its corresponding masked GDPH maps:

```bash
python -m data_preprocess.luad_normal_ablation --candidate A --wsi-list /path/to/LUNG_path/TCGA/all_data.json --initial-feature-dir /path/to/initial_165_maps --available-mask-dir /path/to/available_masks --final-feature-dir /path/to/processed_165_maps --output-dir /path/to/normal_candidate_A
python -m prognosis.train_mfm_msfm --cohort luad --dataset tcga --data-root /path/to/LUNG_path --feature-map-dir /path/to/normal_candidate_A --model macro --output-dir runs/luad_normal_A
```

## Locked external evaluation and final models

Supply a matching external micro CSV from the same fold's selected tiles, then evaluate the fixed MSFM checkpoint:

```bash
python -m prognosis.evaluate_external --cohort hcc --dataset kmmufh --data-root /path/to/HCC_path --feature-map-dir /path/to/external_processed_maps --model fusion --checkpoint runs/hcc_msfm/fold_0/best.pt --micro-features-csv /path/to/external_micro/fold_0.csv --micro-feature-dim 168 --output-dir results/hcc_external_fold_0
```

`external_predictions.csv` contains the unbounded head logit as `risk_score`. The external cohort is never used for MFM/MSFM epoch or checkpoint selection.

For downstream continuous-risk analyses, fit final MFM on all discovery WSIs. Use its checkpoint to make final key tiles, supply corresponding final micro features from the external extractor, then fit final MSFM. `--cv-output-dir` sets training duration to the ceiling of the median optimal epoch from all ten discovery folds:

```bash
python -m prognosis.train_mfm_msfm --stage final --cohort hcc --dataset tcga --data-root /path/to/HCC_path --feature-map-dir /path/to/processed_maps --model macro --cv-output-dir runs/hcc_mfm --output-dir runs/hcc_final_mfm
python -m prognosis.train_mfm_msfm --stage final --cohort hcc --dataset tcga --data-root /path/to/HCC_path --feature-map-dir /path/to/processed_maps --model fusion --micro-feature-dim 168 --micro-features-csv /path/to/final_micro.csv --macro-checkpoint runs/hcc_final_mfm/final.pt --cv-output-dir runs/hcc_msfm --output-dir runs/hcc_final_msfm
python -m prognosis.export_patient_risk --predictions-csv runs/hcc_final_msfm/discovery_predictions.csv --output-csv results/hcc_patient_os_risk.csv
```

The patient-risk exporter requires one WSI per TCGA patient and rejects silent aggregation. `prognosis.endpoints` supplies the prespecified three-year OS subgroup and fixed-OS-risk PFI analyses:

```bash
python -m prognosis.endpoints three-year-os --input-csv /path/to/os_risk_table.csv --output-dir results/os_3year
python -m prognosis.endpoints pfi --risk-csv results/hcc_patient_os_risk.csv --endpoint-table /path/to/TCGA_metadata_Aug2018.xlsx --endpoint-sheet TCGA-CDR --endpoint-id-col bcr_patient_barcode --output-dir results/pfi
```

`prognosis.biomarkers rank` counts discovery top-10 feature attribution frequencies. `prognosis.biomarkers validate` fits a max-logrank cutoff between the discovery 10th and 90th percentiles and transfers it unchanged externally. This is exploratory multiple-testing analysis; record all considered features. Each command has `--help` for its required files.

```bash
python -m prognosis.biomarkers rank --scores-csv /path/to/discovery_feature_scores.csv --top-n 10 --output-csv results/hcc_top10_frequency.csv
python -m prognosis.biomarkers validate --cohort hcc --data-root /path/to/HCC_path --discovery-scores-csv /path/to/discovery_feature_scores.csv --external-scores-csv /path/to/external_feature_scores.csv --scope all --feature-name glcm_Contrast --output-json results/hcc_glcm_contrast.json
```

The reviewer-response clinical and treatment-field disclosure can be regenerated without exporting patient rows:

```bash
python -m prognosis.clinical_availability --cohort hcc --dataset tcga --data-root /path/to/HCC_path --output-json results/hcc_clinical_availability.json
```

`prognosis.clinical_cox` joins fixed risk predictions to categorized clinical covariates and fits risk-only, clinical-only, and combined Cox models on complete cases. It reports in-sample C-indices and hazard ratios; those values are descriptive and must not be reported as locked external performance:

```bash
python -m prognosis.clinical_cox --cohort hcc --dataset tcga --data-root /path/to/HCC_path --predictions-csv runs/hcc_final_msfm/discovery_predictions.csv --categorical Gender ajcc_pathologic_stage --output-dir results/hcc_clinical_cox
```

## Verification and provenance

```bash
python -m pytest -q
python -m compileall -q configs data_preprocess prognosis demo
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/morphx_wheels
git diff --check
```

The models and clinical mappings were checked against the supplied `MorPathFinder` and `MorPathFinder_LUNG` trees. Provenance, source disagreements, and real-data audit findings are in [docs/reproducibility.md](docs/reproducibility.md). This release deliberately omits image feature-extraction code and third-party Pathomics source. Full manuscript-metric reproduction requires the prepared inputs and a real cohort run.

[Project scope and outstanding inputs](docs/project_scope.md) records every main step, the evidence for this code release, and the external components needed for a full paper rerun.
