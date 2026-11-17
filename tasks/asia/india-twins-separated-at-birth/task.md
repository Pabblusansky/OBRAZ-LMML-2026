Task Overview
📖 Storyline
India's monuments have stood for centuries, preserving stories in stone, marble, and intricate architecture. Today, those stories also live inside an ever-growing digital heritage archive containing thousands of photographs collected from across the country, from ancient caves and majestic forts to iconic gateways, temples, palaces, and memorials.

Over time, the archive has become increasingly difficult to organize. Some files originate from the exact same photograph, even though they may no longer appear identical at first glance. At the same time, many different photographs capture the same monument from similar viewpoints, under similar lighting conditions, or only moments apart.

This creates a challenging mix of true duplicates and photographs that are visually similar but represent distinct captures. Your mission is to restore order to the archive by identifying every pair of images that originates from the exact same source photograph, without being misled by different photographs that simply happen to show the same monument.

📌 Task Title
Twins Separated at Birth

🔥 Difficulty
Easy

🎯 Objective
Given a shuffled collection of photographs of Indian monuments, identify all unordered pairs of images that originate from the same source photograph.

Two images showing the same monument are not necessarily duplicates. Different photographs of the same landmark must be treated as distinct images, even when they show similar viewpoints, composition, or lighting. Not every image is guaranteed to have a duplicate.

Recover as many true duplicate pairs as possible while avoiding false matches.

🚩 FLAG
The flag follows the format SIGMOID_<PHRASE> and is revealed only when the submitted solution reaches the required evaluation threshold.

📂 Task Input
Download train-input.zip from the participant data folder.

Participants receive:

train-input.zip — the training images, available through the Google Drive link above;
train-output.csv — the known duplicate pairs, included in this task folder.
The SHA-256 hash of train-input.zip is:

2e1a5df7bd6c87c581dfdd890ff6d74ebd24dad7a4072d333c4212c6f061316a
Two images are duplicates only when they originate from the exact same source photograph. Different photographs of the same monument are not duplicates, even if they appear visually similar.

During official evaluation, the submitted program receives a hidden test input. The expected duplicate pairs remain hidden from participants.

✅ Expected Output
Participants submit a Python program named:

main.py
During evaluation, main.py processes the provided input archive and must create a file named output.csv in the same working directory.

The CSV must contain exactly these columns:

image_a,image_b
Each row represents a pair of images predicted to originate from the same source photograph.

Image pairs are unordered: (image_a, image_b) is equivalent to (image_b, image_a).
Repeated or reversed duplicate rows receive no additional credit.
Self-pairs such as (image_a, image_a) are invalid.
Image names must refer to files contained in the provided evaluation archive.
If no duplicate pairs are predicted, output.csv must still contain the header row.
🏆 Scoring Criteria
The submitted pairs are compared with the hidden expected duplicate pairs. The evaluator calculates precision, recall, and the F1 score.

✅ F1 score greater than or equal to 0.88: 100 points (PASS), and the flag is revealed.
❌ F1 score below 0.88: 0 points (FAIL), and the flag is not revealed.
