# Industrial quality assurance for metal castings

**Module 4, Sprint 2: Digital image processing** | Turing College, Data Science Program

Classify submersible pump impellers as defective or sound, time the algorithm, and box the defect, using classical image processing and classical machine learning. No neural networks.

Dataset: [Kaggle, *casting product image data for quality inspection*](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), `casting_512x512` only: 1,300 unaugmented images, 781 defective and 519 sound. They are committed under `data/raw/`, so nothing needs downloading.

---

## Result

On the sealed test split (520 images, opened once), the shipped SVM gets **518 right: 99.6% accuracy (95% CI 98.6 to 99.9%)**, F1 0.997 for the defective class, one missed defect out of 313 and one false alarm out of 207. The whole product, from JPEG on disk to a decision and boxes, runs at a **median of 141 ms per image** (156 ms at the 95th percentile) on one core.

Validation, which chose every setting, scored lower: 96.4% accuracy with 14 false alarms on 156 sound castings. That is the more conservative expectation for new data.

| Approach, fitted on explore, scored on validation | Accuracy | Defect F1 |
|---|---|---|
| Background floor: random forest on workbench pixels only | 0.846 | 0.873 |
| Rule-based weighted score, five named features | 0.718 | 0.810 |
| Random forest, 36 part features | 0.959 | 0.967 |
| **SVM with RBF kernel, 36 part features (shipped)** | **0.964** | **0.971** |

## Two data problems, measured before modeling

**The same castings were photographed repeatedly:** 721 of the 1,300 images belong to 249 castings photographed two to 32 times at different rotations. Exact hashing cannot see this, so near-duplicates are found with a rotation-free signature: the rim unwrapped to polar coordinates, stripped of everything every casting shares, and compared at the best rotation with an FFT. A plain stratified split would have put a photograph of the same casting as 128 of the 520 test images into the training data. The split used here is group-aware (`StratifiedGroupKFold`), keeps each casting in one partition, and still lands on 390 / 390 / 520.

**The classes were photographed under different light:** defective castings sit on a darker workbench. Corner patches containing no metal separate the classes with an AUC of 0.77, and a model on background pixels alone reaches 0.846 validation accuracy. Every model feature is therefore computed strictly inside the part and normalized by the part's own brightness. An exposure-matched control changes only one thing: it compares random balanced subsets with subsets where background brightness is equal between the classes. On the matched subsets the background probes fall from 0.776 to 0.566 accuracy, most of the way to chance, while the part features hold at 0.983. The classifier reads metal, not the lamp.

## Pipeline

```
load as 1 channel   the files are RGB JPEG with three identical planes; 786 KB -> 262 KB
median 3x3          discards isolated grain and JPEG specks while keeping a step edge a step
find the cavity     darkest large object; gives the center and the radius everything is measured in
rim band            1.15 to 2.05 cavity radii, the machined surface; nothing outside it is a feature
normalize           divide by the band's own median, which cancels a change of exposure
isolate             fill outside the band before any operator with a spatial footprint runs
black top-hat       closing minus image at radii 5, 11, 21: how much darker than the metal around it
polar unwrap        rotation becomes a cyclic shift, so angular statistics are rotation-invariant
```

Features come in six families, all inside the band: top-hat response, Sobel gradient, normalized intensity, angular wedge statistics, rotation-invariant uniform LBP, and response areas and blobs.

## Localization

The top-hat also responds to structure every casting has: the groove beside the cavity and the outer chamfer. Boxes are therefore drawn where the response exceeds what **sound** exploration castings produce at the same radius. The reference quantile and the minimum area were chosen on the exploration split to box the most defective castings while boxing at most 10% of sound ones. On test, 9.2% of sound castings and 47.3% of defective castings get a box (validation: 7.7% and 64.1%), and the median box is about 20 px across, a spot on the rim. Damage outside the band, such as burrs and flash past the part's edge, is not boxed, and the dataset has no location labels to score boxes against.

## Iterations that changed the design

| First design | Measurement | Replacement |
|---|---|---|
| Otsu threshold to find the part | largest component covers 36 to 80% of the frame | dark vane cavity as anchor, found in 390/390 images |
| OR of per-feature alarms | 52% recall on validation | weighted score with one threshold |
| Plain disk top-hat | 1,412 ms per image | decomposed octagon, 87 ms, 1.3% mean pixel change |
| Boxes on the raw response | boxes on every casting, around the cavity | per-radius sound-metal reference |
| Plain stratified split | 128 test images with a twin in training | group-aware split, 0 twins across partitions |

## Robustness

The shipped model was re-run on 120 validation images after controlled changes. A gain of ±10% flips 2 decisions and a 90° rotation flips none. A 10° rotation flips 13 and JPEG quality 50 flips 39, almost all toward false alarms. Photographs must reach the classifier at the rig's native resolution and compression.

## Repository structure

```
.
├── data/
│   ├── raw/{def_front,ok_front}      <- 781 + 519 JPEGs, the brief's casting_512x512
│   └── processed/                    <- features, results and duplicate groups, written by the notebook
├── notebooks/
│   └── casting_quality_inspection.ipynb
├── src/casting_qa/
│   ├── config.py                     <- every constant, threshold, grid and sample size
│   ├── dataset.py                    <- inventory, audit, cleaning, group-aware 30/30/40 split
│   ├── duplicates.py                 <- rotation-free near-duplicate detection and grouping
│   ├── geometry.py                   <- cavity anchor, rim band, polar unwrapping
│   ├── features.py                   <- the pipeline stages and every feature
│   ├── localize.py                   <- sound-metal reference, boxes, end-to-end inspection
│   ├── robustness.py                 <- controlled perturbations of a photograph
│   ├── plots.py                      <- every figure, returns a Figure
│   └── modeling/classify.py          <- rules, classifiers, intervals, McNemar, timing, controls
├── tests/                            <- 68 tests, one file per module, on synthetic castings
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml
└── uv.lock
```

## Tests

68 tests run on synthetic impellers rather than the real dataset, so a defect can be planted at a known coordinate. The ones that guard the conclusions:

- **`test_no_feature_can_see_the_background`** repaints the background and requires every model feature to stay put.
- **`test_nuisance_probes_never_read_the_casting`** repaints the part and requires every background probe to stay put, so the background floor really is background-only.
- **`test_a_rotated_photograph_of_the_same_casting_matches_and_another_casting_does_not`** pins the near-duplicate detector.
- **`test_the_planted_defect_is_boxed_where_it_was_drawn`** and **`test_a_sound_casting_gets_no_strong_box`** pin the localizer.
- **`test_grouped_split_keeps_every_group_in_one_partition`** pins the leakage fix.

## How to run

```bash
uv sync --extra dev
uv run jupyter lab notebooks/casting_quality_inspection.ipynb    # select the "Python 3 (ipykernel)" kernel
```

Or execute end to end from the command line (about 4 minutes on 10 cores):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/casting_quality_inspection.ipynb
```

Checks:

```bash
uv run pytest -q        # 68 tests
uv run ruff check .     # package, tests and notebook
```

## What to improve next

- Grouped cross-validation over all 1,300 images, to replace one split's error rates with a mean and a spread.
- One photography session for both classes, which removes the lighting confound at the source.
- Defect-location labels on a few hundred images, so boxes can be scored and the band widened to the part's edge.
- Augmentation with small rotations, tone curves and JPEG re-encoding, then re-running the robustness check.
- A roundness check on the cavity anchor to reject tilted photographs before they become false alarms.
