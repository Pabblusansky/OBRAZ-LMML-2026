# Albania – Arată a anomalie! (Mediu)

Streaming anomaly detector for bunker sensors. Implement `class Detector` with `update(reading: dict) -> float` in `detector.py` (score in [0,1], threshold 0.50).
Score = 100 × (0.80·F1 + 0.05·point recall + 0.075·drift recall + 0.075·collective recall); ≥70 releases the flag `SIGMOID_<WORD>`.
Runtime: Python 3.11, numpy 2.2.4, pandas 2.2.3, scikit-learn 1.6.1, CPU, 60 s, no internet.
Fields: timestamp, bunker_zone, temperature_c, humidity_pct, vibration_g, acoustic_db, air_pressure_kpa, co2_ppm.
Data: files/train-input.csv, files/train-output.csv (3600 rows, 1-min steps).
