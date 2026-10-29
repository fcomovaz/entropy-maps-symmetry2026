# MVTEC-AD Daaset cleaning and converting

## Issue

* The main issue with this file is that all images are in RGB format and they have diferent sizes. This is an issue to consume in an easy way the dataset. so We need to convert to grayscale.
* Second issue is some images can be have diferent sizes, and some have a big resolution or a low resolution. Through a previous analysis we noticed minimum size as 128x128 to apply to all images. In this case can be possible or there is a possibility that the image is not a perfect square, it can be a rectangle, in this case we won't resize to square 128px due to image ratio change, so we need to do letterboxing/padding and make the larger side 128px and fill the rest with 0's (black), in this way we preserve the image ratio.

### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\raw\mvtec-ad`

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.

## Proceeding

- [] Read AGENTS.md file to know the general rules of the project.
- [] Analyze the request and create a plan
- [] Update the configuration file needed (it can be the base.toml or create an specific one for this task on the images)
- [] Update the `## Plan (AI reviewed and updated)` section to implement post user review, be very specific in this section since another agent would read it and implement it so it needs to be very clear.

## Plan (user provisional)

- [] Convert all images to grayscale
- [] Resize all images and do letterboxing/padding at 128px
- [] Create a new folder with the new structure under the data\preprocessed\mvtec-ad folder (create if not exist)
- [] The preprocessed folder should contain the same structure of folders in the raw, but it's expected to be smaller (due to the size and channel reduction) and contain only the preprocessed images.

## Plan (agent reviewed and updated)