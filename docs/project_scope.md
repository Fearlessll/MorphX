# Project scope and outstanding inputs

This release contains the MorphX model and analysis code shared by HCC and LUAD. WSI feature extraction and the third-party Pathomics source are excluded. The commands in `README.md` start from prepared maps and micro-feature tables. The table below separates code coverage from evidence that manuscript numbers have been reproduced.

| Step | Entry point | Verification and remaining input |
| --- | --- | --- |
| WSI segmentation and image feature extraction | External, not included | Supply reproducibly prepared tissue, texture, nuclear/cell maps and records of the upstream tools, weights, and parameters. |
| Tissue + texture + nuclear concatenation and 256-pixel maps | `data_preprocess.concat_feature_maps`, `data_preprocess.normalize_feature_maps` | Channel dimensions and geometry format checked. Newly normalized coordinates differ from undocumented historical transforms. |
| Fixed discovery ten-fold MFM | `prognosis.train_mfm_msfm --model macro` | Synthetic HCC/LUAD folds ran. All four real-cohort inputs passed read-only shape and clinical-match preflight; no real GPU ten-fold rerun. |
| IG attribution and key tiles | `prognosis.key_tiles` | Synthetic IG and top-K selection ran. A geometry JSON from the same map construction is required for each slide. |
| Top-K micro features | External, not included | Supply fold-specific 168/165-column CSVs using the model-selected tile manifests. The model loader checks row names, dimensions, and finite values. |
| Fixed discovery ten-fold MSFM | `prognosis.train_mfm_msfm --model fusion` | Synthetic training ran with clearly labeled mock micro vectors; real fold-specific micro tables must be generated before manuscript reproduction. |
| Locked external evaluation | `prognosis.evaluate_external` | Synthetic HCC and LUAD external runs passed. Use KMMUFH or the 301-slide GDPH paper list without checkpoint selection on external data. |
| Final all-discovery model and continuous risk | `--stage final --cv-output-dir`, `prognosis.export_patient_risk` | Synthetic final MFM fit passed. Real final MFM and MSFM have not been trained in this release. |
| Endpoints, feature attribution, clinical disclosure | `prognosis.endpoints`, `prognosis.biomarkers`, `prognosis.clinical_availability`, `prognosis.clinical_cox` | Interface and synthetic clinical fit checked; actual patient-level PFI inputs, selected clinical covariates, and statistical outputs remain external. |
| Reviewer LUAD candidate-normal ablation | `data_preprocess.luad_normal_ablation`, then MFM CV/external commands | Candidate masks and QC fields tested on synthetic arrays; real maps need run and review. |
| Manuscript comparator models | Pinned third-party source and weights required | CLAM, TransMIL, GigaPath, and BEPH/BETH are outside the MorphX implementation; their benchmark rows have not been regenerated here. |

## Resolved training choice

The author selected standard Cox partial likelihood for this project on 2026-09-19. It is now the default for all MFM/MSFM training. The original HCC/LUAD scripts' two-term modified loss remains available only as an explicit historical option. Earlier checkpoints and manuscript numbers cannot be relabeled as Cox results; trace their provenance or retrain and regenerate the analyses before claiming numerical agreement.

## Issues requiring an author decision before claiming exact paper reproduction

1. The fixed LUAD ten-fold split covers 280 of 285 discovery slides as validation cases. The code preserves the split. State the five never-validated slides' role in the reported cross-validation and final training.
2. Two KMMUFH identifiers used in the 151-slide list have duplicate clinical rows; one pair conflicts. The loader follows the original first-row rule with a warning. A data curator must confirm the correct endpoint before presenting an audited external result.
3. Historical 256-pixel maps lack the inverse geometry needed for exact WSI tile coordinates. Produce new maps plus geometry, or supply a verified transform for the historical normalization.

The synthetic demo is an installation and interface check. Its risk scores, metrics, and mock micro features are not manuscript evidence.
