"""Walu Season: UCB1 Strategy for Multi-Armed Bandit.

Uses Upper Confidence Bound (UCB1) with exploration constant c=4.5 to balance
searching for the high-reward Walu school against exploiting known spots.
"""

import json
import math
import sys


class Strategy:
    def __init__(self, n_spots: int, n_casts: int, c: float = 4.5):
        self.n_spots = n_spots
        self.n_casts = n_casts
        self.c = c
        self.casts = [0] * n_spots
        self.total = [0.0] * n_spots
        self.t = 0

    def choose(self) -> int:
        # Initial exploration: try every spot once
        for spot in range(self.n_spots):
            if self.casts[spot] == 0:
                return spot

        # UCB1 selection
        log_t = math.log(self.t)
        best_spot = 0
        best_score = -1e9
        for s in range(self.n_spots):
            mean = self.total[s] / self.casts[s]
            bonus = self.c * math.sqrt(log_t / self.casts[s])
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_spot = s
        return best_spot

    def update(self, spot: int, catch: float) -> None:
        self.t += 1
        self.casts[spot] += 1
        self.total[spot] += catch


def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
