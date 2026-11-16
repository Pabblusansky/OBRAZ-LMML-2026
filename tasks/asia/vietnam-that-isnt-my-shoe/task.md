# Vietnam — That isn't my shoe!

## 📖 Storyline
Vietnam is one of the world's largest athletic-footwear manufacturing hubs: Nike, Adidas and Converse all run major production there. On the night shift at a footwear plant outside Hanoi, a scanner named Sherlock Soles sorts every sneaker into three bins, Nike, Adidas and Converse, until strangers start arriving. A plain classifier is forced to guess and cheerfully drops a Croc into "Adidas." So Sherlock learns to trust only what it can verify. A genuine shoe always carries the real thing, the exact stitch tolerances and sole geometry an authentic Nike never gets wrong, and earns its label. Everything else falls into one honest bucket: unknown. The counterfeit Nike that sits right atop the real ones but drifts on those tiny tells is unknown. The knockoff maker who copies a rival's whole form factor, a shoe shaped like a Converse but honest in none of the details, is unknown. Only the label makes it a label; the rest, however convincing, stays unknown.

## 📌 Task Title
That isn't my shoe!

## 🔥 Difficulty
Medium

## 🎯 Objective
Open-set recognition. Train only on the three known brands, then label each test image as one of them or reject it as the literal unknown. The test set mixes knowns with unseen look-alike sneakers (Puma, Reebok, Vans, Asics).

## 🚩 FLAG
SIGMOID_<WORD>, released by the evaluator only when both scoring gates pass.

## 📂 Task Input
Training data contains the following:

Real 256×256 sneaker photos. train-input/ + train-output.csv: 735 known images (nike/adidas/converse). The test set is held by the evaluator and is not provided. Images are referenced by filename only.

Environment for testing: Python 3.10–3.12 with torch==2.11.0, torchvision==0.26.0, numpy==2.4.6, Pillow==12.3.0.

## ✅ Expected Output
Submit your trained model as a single .pt checkpoint (no code needed — the platform runs it)

## 🏆 Scoring Criteria
Two metrics are computed on the test set:

- Known accuracy: share of known images labelled with the correct brand.
- Rejection rate: share of unknown images labelled unknown.

✅ Known accuracy ≥ 0.85 and rejection rate ≥ 0.70: 200 points (PASS, flag released)
❌ Either metric below its threshold: 0 points (FAIL)
