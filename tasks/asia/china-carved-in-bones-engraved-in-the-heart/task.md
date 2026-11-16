Task Overview
📖 Storyline
A rubbing is a copy made without a camera. Damp paper is pressed onto a carved surface. Ink is dabbed over it. The carved lines stay white. For centuries this was how China copied its stone steles, bronzes and seals.

In 1899 the scholar Wang Yirong is said to have noticed writing on "dragon bones" sold as medicine. The trail led to Yinxu, near Anyang, the last capital of the Shang kings. There, three thousand years ago, diviners carved questions into turtle shells and ox shoulder blades. Will it rain? Will the harvest be good? Is the king's toothache a curse? Heat was applied until the bone cracked. The cracks were the answer.

In June 1936 archaeologists opened pit YH127. It held more than 17,000 inscribed pieces: a royal archive, buried whole. This is the oldest Chinese writing we have. Over 4,500 different characters are known. Most still cannot be read.

The bones are too fragile to pass around, so scholars work from rubbings. One archivist in Anyang kept a private collection in eight crates. Each crate had a single glyph painted on its lid and held five bundles of rubbings. Five, as the diviners often put one question to five shells. Nobody was told why the bundles were sorted that way. Colleagues only remember a saying: "The bones answer whoever counts carefully enough."

📌 Task Title
Carved in bones, engraved in the heart

🔥 Difficulty
Medium

🎯 Objective
Each crate keeps one letter. It is written in how often the crate's glyph was carved across its five sets. Gather the letters and form the word.

🚩 FLAG
SIGMOID_<WORD>: one word of eight letters, in uppercase.

📂 Task Input
The dataset holds greyscale JPEG rubbings of inscribed bones and shells.

train/images/: 1,800 labelled rubbings.
train_labels.csv: one row per character, with image, x, y, w, h, glyph_id. Boxes are in pixels from the top-left corner. Uncommon glyphs share the id rare.
glyphs.csv: glyph_id, modern_character, train_boxes.
crates.csv: crate, glyph_id, modern_character. The glyph of each crate.
crates/crate_XX/set_Y/: 1,200 unlabelled rubbings, 30 per set. File names carry no meaning.
baseline_solution.py shows the input and output formats.

✅ Expected Output
Submit on the platform:

The flag in the format of SIGMOID_<WORD>.

A URL with your code, proving the solution, a counts.csv (crate, set, glyph_id, count, one row per set) and a short write-up.

🏆 Scoring Criteria
✅ Correct flag with supporting code, counts and write-up: 200 points (PASS)
❌ Incorrect flag, or a correct flag without the supporting material: 0 points (FAIL)
