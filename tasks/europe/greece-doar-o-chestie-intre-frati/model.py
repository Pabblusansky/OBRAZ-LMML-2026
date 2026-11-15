import torch
import torch.nn as nn

# Ordinea exactă a celor 9 caracteristici numerice așteptate de stratul liniar al modelelor
FEATURE_COLUMNS = [
    "creature_mass_talents",
    "oracle_omen",
    "aggression_index",
    "divine_aura",
    "aegis_resonance",
    "armor_resistance",
    "witness_count",
    "styx_disturbance",
    "omen_intensity",
]


class MythicThreatClassifier(nn.Module):
    """Clasificator binar compact (regresie logistică) comun celor trei modele-oracol."""

    def __init__(self):
        # Inițializăm clasa de bază nn.Module pentru a gestiona parametrii și straturile PyTorch
        super().__init__()
        # Un singur strat liniar care mapează cele 9 caracteristici de intrare la un logit scalar
        self.linear = nn.Linear(len(FEATURE_COLUMNS), 1)

    def forward(self, x):
        # Aplicăm transformarea liniară (x @ W^T + b) și eliminăm ultima dimensiune de mărime 1
        # Rezultatul reprezintă logit-ul brut (înainte de funcția sigmoid)
        return self.linear(x).squeeze(-1)
