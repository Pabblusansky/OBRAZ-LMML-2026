# Ecuador – Mirajul de pe Ciclopaseo (Mediu)

Objective: write main.py that perturbs test-input.png (each RGB value ±1 max) so MobileNetV3-Large
(IMAGENET1K_V2, no resize/crop) predicts class 665 (moped) with softmax >= 0.99. Output test-output.png
(RGB PNG, same size, <= 1 MiB). Runs offline: `python main.py`, 60 s, 2 CPU, 2 GiB.
Flag format: SIGMOID_<WORD>, released by the platform when the submission passes.

Files: files/ (challenge_model.py, baseline_solution.ipynb, train-input.png, train-output.png,
classifier/mobilenet_v3_large-5c1a4163.pth – official, SHA-256 verified).
Full Romanian description was pasted in chat (Paseo Dominical / Ciclopaseo, Quito).
