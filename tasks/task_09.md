# Optimization Entropy Maps


## Issue

- The task is optimize the entropy maps, right now the enropy maps are calculated by iterating N times over the image, where N is the number of patches in the image. One of the solutions you provided was to iterate over the patches in a single pass, what i did understand was to calculate only the "coordinates" of the pathes and then iterate over them in a single pass.
- i want to check if this is true, also i would like another solution type using a c function and call it (at the end this will be migrated to c but not yet). i want a big optimization to reduce the velocity in batch 1 (review REVIEW_01.md in the root of this code folder).
- also if you have another solution to overpass this overhead please tell me, i want a solution to overpass the pytorch time but with less parameters, and also maybe i will increase the patches so overpass this time is necessary.
- you can start by creating another laboratory in understanding doing this optimization, in this way you will never touch the src/ files yet.
- i really want to optimize this stage and this critical operation use any computer science resource you know and remember this will be migrated to c but not yet so if is easy to implement in that language it will be helpful. but the optimization is requiered, in the understanding for this also ceate a script with scientific graphs to see the comparison if there is a real improvement.



## Plan

> Don't execute anything, just plan it

- [] Read AGENTS.md file to know the general rules of the project.
- [] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [] Update this file with the plan, stages and everything to resolve and implement the task.

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.