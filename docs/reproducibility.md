# Reproducibility and source alignment

This repository consolidates the supplied HCC `MorPathFinder` and LUAD `MorPathFinder_LUNG` model code. It starts from prepared morphological feature maps and micro-feature CSVs. WSI segmentation, image feature extraction, and third-party Pathomics source are outside this release.

## Model and method alignment

| Topic | Manuscript/source evidence | Implementation |
| --- | --- | --- |
| Feature layout | HCC 8 + 40 + 120; LUAD 5 + 40 + 120 | One feature-map loader with absolute channel indices in `configs/cohorts.py` |
| Tissue classes | Manuscript prose names 7 HCC and 4 LUAD tissue classes; source maps add an `empty` channel | Model inputs have 8 and 5 tissue channels, respectively |
| MFM widths | HCC residual widths 128/256/512/1024; LUAD 256/512/1024/2048 | One configurable ResNet; historical tensor shapes checked against both source models |
| MSFM head | BatchNorm, dropout, linear 64, linear 32, BatchNorm, ReLU, linear 1 | Same order and dimensions |
| Training and prediction scores | Source optimization consumes a sigmoid score; risk analyses use the head logit | Sigmoid by default in training, logit in predictions |
| Training loss | Source scripts use a two-term modified loss; manuscript displays standard Cox partial likelihood | Author selected standard Cox as the default on 2026-09-19; Breslow ties and per-event mean reduction; historical loss remains an explicit option |
| Optimizer and selection | Event-weighted sampling without replacement, batch 10, Adam, source scheduler; discovery validation selects checkpoints | Same defaults; external data never selects epochs or checkpoints |
| Attribution | 50-step zero-baseline integrated gradients, absolute channel sum, top 16 positions | Same defaults; source-grid positions require matching geometry JSON |
| Final model | Full discovery cohort, median optimal CV epochs | `--stage final --cv-output-dir` uses the ceiling of the median of ten validation-optimal epoch counts |
| MFM activation ablation | A run without sigmoid during training | `--train-output logit`; default remains `sigmoid` |

The old MFM implementation instantiated SelfNorm without using it in the active forward path. The unified model retains those state-dictionary keys so historical MFM checkpoints can load. A new Cox-trained model has a different training protocol from a historical two-term-loss model. Do not relabel historical weights, tables, or figures as standard-Cox results; regenerate dependent outputs before claiming numerical reproduction. Each new fold records its loss, and final training rejects ten-fold metrics with a mismatched or unknown loss.

## Prepared input contract

Provide a JSON WSI list, clinical CSV, and one `256 × 256 × C` HWC NumPy array per listed WSI. `C` is 168 for HCC and 165 for LUAD, ordered tissue, texture, nuclear/cell according to `configs/metadata`. The fixed ten TCGA split files must be supplied unchanged. GDPH uses the 301-slide `all_data_mil.json` list for the manuscript analysis.

The optional `data_preprocess.concat_feature_maps` and `data_preprocess.normalize_feature_maps` commands assemble already computed feature families and generate model maps with geometry records. They do not extract WSI features. Historical 256-pixel maps have no inverse-geometry metadata; do not map integrated-gradient pixels back to WSI coordinates without a verified transform. The supplied `.npy` maps also lack channel-name metadata, so shape checks alone cannot establish semantic channel order.

For MSFM, provide one row per WSI with `wsi_id` and exactly `feature_0` through `feature_167` (HCC) or `feature_164` (LUAD). Separate fold and external tables must correspond to the selected MFM checkpoints and key-tile manifests. The synthetic demo uses mock micro vectors solely to check the interface.

## Real input audit (read-only, 2026-09-19)

| Cohort and dataset | Analysis list | Maps with expected shape | Clinical matches | Split finding |
| --- | ---: | ---: | ---: | --- |
| HCC TCGA-LIHC | 330 | 330 of 330, 256 × 256 × 168 | 330 | All 330 occur once among ten validation folds |
| HCC KMMUFH | 151 | 151 of 151, 256 × 256 × 168 | 151 | Locked external cohort |
| LUAD TCGA-LUAD | 285 | 285 of 285, 256 × 256 × 165 | 285 | 280 occur once; 5 occur zero times as validation cases |
| LUAD GDPH, `all_data_mil.json` | 301 | 301 of 301, 256 × 256 × 165 | 301 | Locked external cohort |

These are shape and linkage checks, not validation of every clinical field against source records. HCC TCGA has six zero-day censored observations, retained as in the source protocol. KMMUFH has two duplicated identifiers used by the 151-slide list, including one conflicting survival record. The loader warns and follows the source script's first-row rule; the conflicting record needs curator review before an audited external result. The LUAD split leaves five discovery slides without a validation appearance; the code preserves the supplied files and reports this limitation.

The LUAD candidate-normal-region MFM ablation is implemented in `data_preprocess.luad_normal_ablation`. `prognosis.clinical_availability` reports covariate availability without patient rows. `prognosis.clinical_cox` fits descriptive in-sample risk-only, clinical-only, and combined models; its numbers are not locked external performance.

The manuscript's CLAM, TransMIL, GigaPath, and BEPH/BETH comparisons depend on separately pinned third-party code, encoder weights, and feature-bag manifests. This MorphX repository does not claim to reproduce those benchmark rows. A full paper-metric rerun still requires the prepared maps, tile micro features, clinical endpoints, fixed folds, and GPU training of all MFM/MSFM folds and final models.
