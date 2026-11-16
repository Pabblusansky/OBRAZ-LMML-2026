# Mirajul de pe Ciclopaseo

| Field | Value |
|---|---|
| Country | Ecuador |
| Difficulty | Medium |
| Category | adversarial examples — targeted L∞ attack |
| Solved by | Dragos |

## Task
Change every RGB value of `test-input.png` by **at most ±1** so that MobileNetV3-Large
(ImageNet) sees a mountain bike as class 665 (*moped*) with softmax ≥ 0.99. The run is offline,
within 60 s on 2 CPUs.

## Approach (`main.py`)
A targeted MI-FGSM attack: take the gradient of the loss towards "moped" with respect to the
pixels, and step pixels by the sign of the gradient with momentum, clipped to the ±1/255 box.
`run/` holds robustness checks on cropped, mirrored and resized variants.
