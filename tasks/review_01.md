# Reviewing current state - November 11st, 2026

We have been doing all this stuff and we have a lof of stuff in here.

### Research Overview: ($x_{94}$) Descriptor vs. Lightweight CNNs

#### **Hypothesis**
Extracting a static, compact, and immutable descriptor ($x_{94}$) based on multiscale entropy and spatial asymmetry can achieve predictive performance equivalent or superior to that of a lightweight Convolutional Neural Network (CNN) in anomaly detection, while drastically reducing parameter count (to fewer than 100) and computational load.

---

#### **Methodology**
* **Preprocessing**: Letterbox resizing to 128×128 pixels in grayscale (along with their inverted intensity versions).
* **Descriptor Construction (($x_{94}$))**:
  * **85 Shannon entropy features**: Computed across a spatial pyramid (1×1, 2×2, 4×4, and 8×8 grid levels).
  * **9 spatial asymmetry residues**: Geometric comparison (Left-Right, Top-Bottom, and 180° Rotation) between sub-patches.
* **Experimental Evaluation**: ($x_{94}$) was extracted from the **MVTec-AD (Bottle)** and **KolektorSDD2** datasets. Lightweight classifiers (Logistic Regression, Random Forest, KNN, Decision Tree) were trained and benchmarked against **7 lightweight CNN architectures** (ranging from 1.1k to 1.5M parameters).

---

#### **Key Findings**

1. **Hypothesis Confirmation on Structural Anomalies (MVTec-AD Bottle)**:
   * **Logistic Regression with ($x_{94}$) (`lr (x94)`)** achieved a **ROC-AUC of 0.9231** and an **F1-Score of 0.8000** using only **95 parameters** with an inference latency of **2.29 ms**.
   * It outperformed several CNNs containing tens of thousands of parameters, such as `mobilenet_like` (0.9047), `ds_cnn` (0.8980), `tiny_cnn` (0.8896), and `ultra_tiny` (0.7709).

2. **High Speed on CPU Without Dedicated GPU**:
   * **`lr (x94)`** achieves an inference latency of **2.29 ms to 2.89 ms per image on CPU**, processing between **350 and 2,000+ images per second** (burst throughput).
   * Compared to the top-performing CNN (`small_cnn`, 5.56 ms), ($x_{94}$) is **more than twice as fast** in inference while utilizing **1,000× fewer parameters** (95 vs. 97,650).

3. **Limitations on Micro-Cracks (KolektorSDD2)**:
   * Deeper CNNs (`tiny_resnet` with ROC-AUC **0.9219**) outperform `lr (x94)` (**0.8061**).
   * *Root Cause*: For ultra-thin cracks (1–2 px wide), computing entropy over a 16×16 px grid cell dilutes the defect signal, whereas 3×3 convolutional filters in CNNs retain fine-grained spatial gradients.


### Next Steps

* I need you to go throught the whole repo and check if everything matches, I think it isok but just check it and generate in the root of the repo a markdown file called REVIEW_01.md, this is our first checkpoint, everything looks as the checks and experiments demonstrate the hypothesis we wanted to defend but I'm not sure everything is up-to-date.
* please give me at the end the go ahead if we can continue with a c/c++ implementation, but only if doing the whole checks and everything is consistent and the experiments validates the hypothesis.