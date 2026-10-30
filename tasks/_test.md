# MVTEC-AD Dataset CSV conversion

## Issue

* The main issue addressed in this file is we have the preprocessed images in `data/preprocessed/mvtec-ad` and we need to create a CSV file in `data/csv/mvtec-ad.csv`. The csv must have the following columns:
    - `object_name`: the object name, e.g. [`grid`, `screw`, `zipper`, `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`], one of the 15 categories in the mvtec-ad dataset.
    - `split`: the split, e.g. [`train`, `test`, `ground_truth`], one of the three splits in the mvtec-ad dataset.
    - `type`: the type, e.g. [`good`, `defect`], one of the two types in the mvtec-ad dataset, in the case of defect, the test and ground truth splits are always deffects so we only consider these as one type, same as train is always good.
    - `px_XXXXX`: where XXXXX is the number of pixel from 00000 to (128x128-1), this columns will contain the image as a flat vector, so px_00001 to px_(128x128-1) will have the grayscale value of the pixel.
    - `width`: the width of the image in pixels.
    - `height`: the height of the image in pixels.


### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\mvtec-ad` or
`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\mvtec-ad-inverted`

both path are valids so i can show if use the inverted or the not inverted ones

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [] Read AGENTS.md file to know the general rules of the project.
- [] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
