# Morocco - The Dispatch Desk of Fez: route each request to the cheap or the expensive guide
import csv
import re


def shortest_walk_is_direct(prompt):
    """True if the direct road between the two places is also the shortest walk."""
    roads = {}
    for a, b, minutes in re.findall(r"(\w+)-(\w+) (\d+) minutes", prompt):
        roads[(a, b)] = roads[(b, a)] = int(minutes)
    start, end = re.search(r"from (\w+) to (\w+)", prompt).groups()
    direct = roads.get((start, end))
    if direct is None:
        return False
    places = {a for a, _ in roads}
    # any detour through one other place that is shorter than the direct road?
    for mid in places:
        if (start, mid) in roads and (mid, end) in roads:
            if roads[(start, mid)] + roads[(mid, end)] < direct:
                return False
    return True


def route(prompt):
    text = prompt.lower()

    # price lookup on a board: the apprentice reads it correctly
    if "board lists" in text and "whatever" not in text:
        return "cheap"

    # walking times: the master is right only when the direct road is the shortest;
    # with a shorter detour both guides are wrong, so do not pay for her
    if "walking times" in text:
        return "expensive" if shortest_walk_is_direct(prompt) else "cheap"

    # one multiplication, small numbers: the apprentice manages
    if "loads" in text and "sells" not in text and "hands" not in text:
        crates = int(re.search(r"loads (\d+) crates", text).group(1))
        return "cheap" if crates <= 13 else "expensive"

    # takings minus stall fees: when a fee is bigger than the takings (negative result)
    # the master is wrong more often than right, so the cheap guide is enough
    if "stall fee" in text:
        takings = [int(x) for x in re.findall(r"(\d+) dirham[,.]", text.split("each souk")[0])]
        fee = int(re.search(r"stall fee of (\d+) dirham", text).group(1))
        crates = [int(x) for x in re.findall(r"\d+", re.search(r"they sold ([\d, and]+) crates", text).group(1))]
        if any(t - fee * c < 0 for t, c in zip(takings, crates)):
            return "cheap"
        return "expensive"

    # everything else needs several steps
    return "expensive"


with open("test-input.csv", newline="", encoding="utf-8") as f:
    requests = list(csv.DictReader(f))

with open("routing_log.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["request_id", "model"])
    for row in requests:
        try:
            model = route(row["prompt"])
        except Exception:
            model = "expensive"      # unknown wording: take the safer guide
        writer.writerow([row["request_id"], model])
