# Synthetic CPU demo

Run `python demo/run_synthetic_demo.py`. Output goes to ignored `demo/output/` by default; use `--output-dir /tmp/morphx-demo` for another location.

The script creates 168-channel random maps, synthetic HCC clinical records, one discovery split, and mock 168-feature micro rows. It trains one MFM fold, computes two-step integrated gradients and four key-tile positions for one synthetic slide, evaluates a fixed MFM checkpoint, trains and evaluates one MSFM fold, and fits a final MFM on all synthetic discovery slides.

Key outputs:

```text
train/fold_0/best.pt
key_tiles.csv
external/external_predictions.csv
fusion/fold_0/best.pt
fusion_external/external_metrics.json
final_macro/final.pt
final_macro/discovery_predictions.csv
```

The mock micro rows are spatial channel means from random maps. No WSI, Pathomics image extraction, segmentation, clinical discovery, or manuscript performance is represented. Real pipeline commands and data contracts are in the main README.
