import json
import math
import pathlib
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
        for spot in range(self.n_spots):
            if self.casts[spot] == 0:
                return spot

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


def get_input_path() -> pathlib.Path:
    candidates = [
        pathlib.Path("test-input.json"),
        pathlib.Path(__file__).resolve().parent / "test-input.json",
        pathlib.Path("train-input.json"),
        pathlib.Path(__file__).resolve().parent / "train-input.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return pathlib.Path("test-input.json")


def main():
    input_path = get_input_path()
    with open(input_path) as f:
        rules = json.load(f)

    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            line = sys.stdin.readline()
            if not line:
                return
            catch = float(line)
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
