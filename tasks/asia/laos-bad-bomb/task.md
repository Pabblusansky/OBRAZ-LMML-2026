Task Overview
📖 Storyline
Decades after historical wars in Southeast Asia, farmers in Laos are still finding old artillery shells buried in their fields. A local charity is using cameras to take close-up pictures of the metal objects they dig up.

Most of the shells are completely empty and safe, but about 2% of them have specific defects showing they might still be live and dangerous. The charity needs to find these dangerous shells fast so they can safely destroy them. But there is a catch: the images are completely unlabeled. You need to build a program that looks at the pictures and flags the odd, defective shells so the safety crew knows exactly which ones to handle with care.

📌 Task Title
Bad BOMB

🔥 Difficulty
Medium

🎯 Objective
You are provided with the images/ dataset folder, in which are pictures with shells. Based on the dataset train a model that correctly identifies 28 defective shells.

🚩 FLAG
Format: SIGMOID_<WORD>. The flag is not hidden in the dataset: the evaluator releases it once your submission passes the Scoring Criteria below.

📂 Task Input
The provided data contains:

images/ — a folder with a set of 1125 images 512x512 png of missile shells, in different lighting settings and background settings. Between these shells there are hidden 28 with defects.
✅ Expected Output
Submit a txt file named test-output.txt with a list of 28 defective shells. Use train-output.txt as reference for how to format final ouput.

🏆 Scoring Criteria
Primary metric: exact-match accuracy on the 28 ground-truth anomalous images.
The accuracy is calculated by formula len(output ∩ ground-truth)/len(ground_truth).
✅ Accuracy ≥ 70%: 200 points (PASS)
❌ Accuracy < 70%: 0 points (FAIL)
