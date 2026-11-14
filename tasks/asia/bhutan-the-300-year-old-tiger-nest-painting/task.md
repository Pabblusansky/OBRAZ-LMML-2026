Task Overview
📖 Storyline
Paro Taktsang, also known as the Tiger's Nest, clings to a cliff face nearly 900 meters above the Paro valley, is only reachable by a switchback trail through pine forest and lines of prayer flags. Bhutanese legend holds that in the 8th century, the tantric master Guru Padmasambhava flew to this cliff on the back of a tigress, said to be his consort Yeshe Tsogyal transformed to subdue a demon troubling the Paro valley. Guru Padmasambhava then meditated for three months in the cave the monastery was later built around. Stand at the viewpoint across the gorge and it's easy to believe it: the temple seems to hang in the clouds, gold rooftops catching the light, as if cut right from the essence from the mountain.

A restoration team recently uncovered a painted scroll in the monastery's archive room - a detailed depiction of Taktsang itself, monks seated at the cliff-edge viewpoint gazing across at the temple, bordered with a strip of hand-lettered verse and a red seal in the corner. Conservators date the silk backing to roughly three centuries old, its inscriptions recording the monastery's founding years, when Bhutan was still being unified under the Zhabdrung's rule and rival factions each kept their own account of how events unfolded. Records from that era are notoriously incomplete: entire chronicles were rewritten, discredited, or quietly disappeared whenever a new abbot or governor wanted the official history to read differently.

That's exactly what the monastery's librarian now suspects happened here. Cataloguing notes from decades ago mention the scroll clearly, along with a name that was later scratched out. That name was one of a monk, entrusted with the archive, whose own faction came out on the losing side of one of those old disputes. Rather than let the painting's inscriptions stand as a record that contradicted his lineage's preferred version of events, he appears to have cut the scroll into vertical strips by hand, deliberately mixed them out of sequence, and filed them back into storage under the wrong labels, burying the truth in plain sight rather than destroying it outright. He was never seen at the monastery again. The scroll's fragments have sat scrambled in the archive ever since, rediscovered only now that the collection is finally being digitized. Buried in its reconstruction is something else worth recovering: the scroll's border conceals the flag of Bhutan itself, worked into the composition. You've been brought in to write the code that undoes the betrayer's work and puts the true record back together.

📌 Task Title
The 300-Year Old Tiger Nest Painting

🔥 Difficulty
Medium

🎯 Objective
You are provided with a train set of vertical strips representing vertical strips of the original image, and a train output (Download dataset here), that contains the correct order of the strips and the image that should be reconstructed and adheres to the provided correct order. The strips have been shuffled into random order. Your goal is to find a way to reconstruct the original order of the image strips and the image itself. Use Computer Vision techniques on the given image portions and combine them together to form the original image prior to shredding, saving the new order in which the image should be assembled back into its original form.

🚩 FLAG
Format: SIGMOID_<WORD>. The flag is not stated in plain text anywhere else in this document or dataset. It is released automatically by the evaluator once your main.py properly reconstructs hidden test PNG image and test strips order JSON file, thus passing the Scoring Criteria described below.

📂 Task Input
train-input/ folder contains the randomly shuffled portions of the train/demo image.

└── 📁train-input
    ├── strip_001.png
    ├── strip_002.png
    ├── strip_003.png
    ...
    ├── strip_190.png
    ├── strip_191.png
    └── strip_192.png
Filenames are randomized and relabeled. They do NOT carry any positional information. Do not assume strip_001.png is the leftmost strip.

All image strips have the same width, height, are perfectly cut vertically.

train-output/ folder contains two files:

└── 📁train-output
    ├── reconstructed_image.png
    └── reconstructed_order.json
reconstructed_image.png - is the train/demo image that was originally shredded.
reconstructed_order.json - order of the strips to assemble the train/demo reconstructed_image.png.
IMPORTANT: Make sure your main.py script reads the input from a generic folder named - input (do not specify train- or test- in its name).

To run the task you need Python 3.13 with numpy==2.5.3, opencv-python==5.0.0.93, scipy==1.18.1 and scikit-image==0.26.0 installed (all pinned in requirements.txt).

✅ Expected Output
The "reassembly" code in a Python file named main.py;

The reconstructed image in PNG format named reconstructed_image.png saved in the same directory as your reconstruction main.py script;

The correct order of the strips in a file named reconstructed_order.json, saved in the same directory as your main.py script, with the contents represented in the following format:

[
    "strip_047.png",
    "strip_012.png",
    "strip_089.png",
    "strip_003.png",
    "strip_071.png",
    ...
    "strip_055.png"
]
The obtained flag string, as returned by the evaluator once submitted.

🏆 Scoring Criteria
For the evaluation step, your main.py file is used on different input strips and different image than the one you had available. Below are presented the criteria on which your code is evaluated automatically.

Automated similarity evaluation - the evaluator holds the private ground-truth source image and computes the Structural Similarity Index (SSIM) between it and your submitted reconstructed image. For the flag to be released, the evaluator should achieve SSIM $\ge$ 0.90 on your provided reconstructed image.

Order verification - the evaluator also compares your reconstructed_order.json against the private ground-truth ordering (a full left-right mirror of the correct order is accepted as equivalent). This is used mainly to detect how a high-similarity score was reached - e.g., catching a submission that scores well on SSIM by chance rather than genuine reconstruction. For the flag to be released, the evaluator should achieve a match of correct order $\ge$ 0.90 on your provided reconstructed image strips order.

Execution time limit - your "reassembly" code is run with a hard timeout of 20 seconds. Exceeding it results in an automatic fail for that submission, regardless of partial output. This is done to block exhaustive search strategies/brute-force submissions.

✅ All three criteria above are met: 200 points (PASS), flag released
❌ Any criterion is missed: 0 points (FAIL)
