# Data dictionary

## Source

Kaggle, [*casting product image data for quality inspection*](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (`ravirajsinh45`). Only the `casting_512x512` directory is used, as the brief specifies: 1,300 images without augmentation, against 7,348 augmented ones in the sibling `casting_data` folder, which is not part of this repository.

Images are top-down photographs of submersible pump impellers taken on a fixed rig with a dedicated lighting arrangement.

## Files

| Folder | Images | Label |
|---|---|---|
| `data/raw/def_front` | 781 | defective (1) |
| `data/raw/ok_front` | 519 | sound (0) |

Filenames follow `cast_{def,ok}_0_<id>.jpeg`. The numeric id is not sequential and carries no meaning; it is not used.

## Format

| Property | Value | Note |
|---|---|---|
| Dimensions | 512 x 512 | verified on all 1,300 |
| Stored mode | RGB JPEG | the dataset description says grayscale |
| Actual content | grayscale | the three planes are bit-identical on all 1,300 files |
| Intensity range | 0 to 255 | |
| Compression | lossy JPEG | |

The three-channel storage carries no extra information, so the loader collapses to one channel: 786 KB to 262 KB per image in memory.

## Quality findings

| Finding | Evidence | Treatment |
|---|---|---|
| No corrupt, blank or off-size files | the audit opens every file | none needed; the checks stay as guards |
| No identical pixel duplicates | SHA-256 of decoded pixels | none needed |
| 249 castings photographed 2 to 32 times, 721 images in total | rotation-free detail similarity >= 0.6, above the 99.9th percentile of all pairs; 1 of 637 matched pairs crosses labels | kept; the group-aware split holds each casting in one partition |
| One matched pair with different labels | `cast_def_0_6173` and `cast_ok_0_3064`, similarity 0.62 | kept; either a false match or a label error, grouped together at no cost |

## Derived quantities

| Name | Meaning |
|---|---|
| `anchor.center` | centroid of the dark vane cavity, the part's center |
| `anchor.radius` | equivalent radius of that cavity, about 101 px; every other radius is in these units |
| rim band | annulus from 1.15 to 2.05 cavity radii, the machined surface that is inspected |
| background | everything beyond 2.3 cavity radii, used only by the confound probes |
| response map | black top-hat at radii 5, 11, 21, bright where the surface is locally pitted |
| polar unwrapping | the rim band resampled to 360 angles x 64 radii |
| duplicate group | connected component of the "same casting" graph, in `data/processed/duplicate_groups.csv` |

## Labels

The folder is the ground truth. There is no label file to join against. Defect types in the source include blow holes, pin holes, shrinkage, burrs and chip-offs, but the dataset does not say which image has which, and it does not mark where the defect is.

## Known confound

Defective images were photographed against a darker background than sound ones. On the exploration split, corner patches containing no casting separate the classes with an AUC of 0.767, and a random forest on three background-only probes reaches 0.851 validation accuracy. Every model feature is therefore computed strictly inside the part, and an exposure-matched control quantifies what remains. See the notebook, sections 6 and 11.
