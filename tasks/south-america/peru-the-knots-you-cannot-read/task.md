# Peru – The Knots You Cannot Read (Medium)

Audit brief: confirm the model is what it claims to be. It classifies correctly... It is also carrying something that no input you can give it will ever make it say.

Objective: Confirm the model does what it claims, then find the content it carries that no input can surface, and recover it.
Flag: SIGMOID_<WORD> — single uppercase ML term. Not stored as text in the artifact.
Files: model.safetensors (char-level classifier, ASCII bytes -> 3 registers), corpus.csv (2400 records).
