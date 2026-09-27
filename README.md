# Industrial quality assurance for metal castings

**Module 4, Sprint 2: Digital image processing** | Turing College, Data Science Program

Classify submersible pump impellers as defective or sound, time the algorithm, and box the defect, using classical image processing and classical machine learning. No neural networks.

Dataset: [Kaggle, *casting product image data for quality inspection*](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), `casting_512x512` only: 1,300 unaugmented images, 781 defective and 519 sound. They are committed under `data/raw/`, so nothing needs downloading.

---

## Result

On the sealed test split (520 images, opened once), the shipped model gets **517 right: 99.4% accuracy (95% CI 98.3 to 99.8%)**, F1 0.995 for the defective class, 2 missed defects out of 313 and 1 false alarm out of 207. The whole product, from JPEG on disk to a decision and boxes, runs at a **median of 149 ms per image** (160 ms at the 95th percentile) on one core.

Group-aware cross-validation over the 780 development images puts the expected accuracy at 99.1% (sd 1.6%) and the false-alarm rate at 1.3%, with single folds ranging from 0% to 9.7%, so validation (3.8%) and test (0.5%) both sit inside the normal spread of one split.

| Approach, fitted on explore, scored on validation | Accuracy | Defect F1 | False alarms / 156 |
|---|---|---|---|
| Background floor: random forest on workbench pixels only | 0.851 | 0.878 | 32 |
| Rule-based weighted score, five named features | 0.718 | 0.810 | 110 |
| Random forest, 36 part features | 0.959 | 0.967 | 14 |
| SVM with RBF kernel, 36 part features | 0.967 | 0.973 | 12 |
| Random forest + augmentation | 0.969 | 0.975 | 9 |
| **SVM with RBF kernel + augmentation (shipped)** | **0.979** | **0.983** | **6** |

## Two data problems, measured before modeling

**The same castings were photographed repeatedly:** 721 of the 1,300 images belong to 249 castings photographed two to 32 times at different rotations. Exact hashing cannot see this, so near-duplicates are found with a rotation-free signature: the rim unwrapped to polar coordinates, stripped of everything every casting shares, and compared at the best rotation with an FFT. A plain stratified split would have put a photograph of the same casting as 128 of the 520 test images into the training data. The split used here is group-aware (`StratifiedGroupKFold`), keeps each casting in one partition, and still lands on 390 / 390 / 520. Every cross-validation inside the development data is group-aware as well.

**The classes were photographed under different light:** defective castings sit on a darker workbench. Background pixels containing no metal separate the classes with an AUC of 0.77, and a model on them alone reaches 0.851 validation accuracy. Every model feature is therefore computed strictly inside the part and normalized by the part's own brightness. An exposure-matched control changes only one thing: it compares random balanced subsets with subsets where background brightness is equal between the classes. On the matched subsets the background probes fall from 0.775 to 0.477 accuracy, which is chance, while the part features hold at 0.971. The classifier reads metal, not the lamp.

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

Features come in six families, all inside the band: top-hat response, Sobel gradient, normalized intensity, angular wedge statistics, rotation-invariant uniform LBP, and response areas and blobs. The shipped SVM is trained on the exploration images plus four augmented copies of each (rotation to any angle, gamma 0.85 to 1.15, JPEG quality 40 to 90, each applied independently).

## Every design choice, tested against what it replaced

| First design | Measurement | Replacement |
|---|---|---|
| Otsu threshold to find the part | largest component covers 36 to 80% of the frame | dark vane cavity as anchor, found in 390/390 images |
| OR of per-feature alarms | 52% recall on validation | weighted score with one threshold |
| Gaussian prefilter (sigma 1) | 10 missed defects on validation, F1 0.945 | 3x3 median: 1 missed, F1 0.973 |
| Plain disk top-hat | 1,433 ms per image | decomposed octagon, 112 ms, 1.3% mean pixel change |
| Plain stratified split | 128 test images with a twin in training | group-aware split and folds, 0 twins across partitions |
| No augmentation | 12 false alarms; 76 decisions flipped under perturbation | augmentation: 6 false alarms; 7 flipped |
| Boxes on the raw response | boxes on every casting, around the cavity | per-radius sound-metal reference |
| Cavity roundness as a tilt gate | the tilted false alarm is ordinary (0.87 vs 5th percentile 0.83) | rejected; tilt belongs to the camera mount |

## Localization

The top-hat also responds to structure every casting has: the groove beside the cavity and the outer chamfer. Boxes are therefore drawn where the response exceeds what **sound** exploration castings produce at the same radius. The reference quantile and the minimum area were chosen on the exploration split to box the most defective castings while boxing at most 10% of sound ones. On test, 9.2% of sound castings and 47.3% of defective castings get a box (validation: 7.7% and 64.1%), and the median box is about 20 px across, a spot on the rim. Damage past the part's edge is outside the band and not boxed, and the dataset has no location labels to score boxes against.

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
│   ├── dataset.py                    <- inventory, audit, cleaning, group-aware 30/30/40 split, outputs
│   ├── duplicates.py                 <- rotation-free near-duplicate detection and grouping
│   ├── geometry.py                   <- cavity anchor, rim band, polar unwrapping, radial profiles
│   ├── features.py                   <- the pipeline stages, every feature, the ablation switch
│   ├── localize.py                   <- sound-metal reference, boxes, end-to-end inspection
│   ├── robustness.py                 <- controlled perturbations and training augmentation
│   ├── plots.py                      <- every figure, returns a Figure
│   └── modeling/classify.py          <- rules, group-aware tuning, intervals, McNemar, timing, controls
├── tests/                            <- 83 tests, one file per module, on synthetic castings
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml
└── uv.lock
```

## Tests

83 tests run on synthetic impellers rather than the real dataset, so a defect can be planted at a known coordinate. The ones that guard the conclusions:

- **`test_no_feature_can_see_the_background`** repaints the background and requires every model feature to stay put.
- **`test_nuisance_probes_never_read_the_casting`** repaints the part and requires every background probe to stay put, so the background floor really is background-only.
- **`test_grouped_splits_keep_groups_whole_and_never_score_augmented_rows`** pins the leakage fix inside cross-validation.
- **`test_a_rotated_photograph_of_the_same_casting_matches_and_another_casting_does_not`** pins the near-duplicate detector.
- **`test_the_planted_defect_is_boxed_where_it_was_drawn`** and **`test_a_sound_casting_gets_no_strong_box`** pin the localizer.

## How to run

```bash
uv sync --extra dev
uv run jupyter lab notebooks/casting_quality_inspection.ipynb    # select the "Python 3 (ipykernel)" kernel
```

Or execute end to end from the command line (about 6 minutes on 10 cores; parallel steps leave one core free):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/casting_quality_inspection.ipynb
```

Checks:

```bash
uv run pytest -q        # 83 tests
uv run ruff check .     # package, tests and notebook
```

## What to improve next

- Photograph both classes in one session, which removes the lighting confound at the source.
- Fix the camera mount so the part is always seen from directly above; the one false alarm is a tilted photograph.
- Label defect locations on a few hundred images, so boxes can be scored and the band widened past the part's edge once the background carries no confound.
- Set the decision threshold from the real cost of a missed defect against a re-inspection; the notebook's operating-point table shows the trade.
- Validate on a new production batch before trusting these rates beyond this rig.
