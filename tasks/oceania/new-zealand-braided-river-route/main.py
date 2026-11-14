import heapq
import pathlib
import sys

try:
    from environment import load_map, ACTIONS, FORD, WALL, WATER, is_bridge, is_winch
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from environment import load_map, ACTIONS, FORD, WALL, WATER, is_bridge, is_winch


def bridge_to_bit(valley):
    sorted_bridges = sorted(valley.bridges)
    return {b: 1 << i for i, b in enumerate(sorted_bridges)}, sorted_bridges


def get_initial_mask(valley, b2bit):
    mask = 0
    for b in valley.initial_open:
        mask |= b2bit[b]
    return mask


def solve(valley):
    b2bit, sorted_bridges = bridge_to_bit(valley)
    init_mask = get_initial_mask(valley, b2bit)

    winch_mask_toggle = {}
    for r in range(valley.height):
        for c in range(valley.width):
            sym = valley.cell(r, c)
            if is_winch(sym) and sym in valley.winches:
                toggle = 0
                for b in valley.winches[sym]:
                    toggle |= b2bit[b]
                winch_mask_toggle[(r, c)] = toggle

    bridge_cell_mask = {}
    for r in range(valley.height):
        for c in range(valley.width):
            sym = valley.cell(r, c)
            if is_bridge(sym):
                bridge_cell_mask[(r, c)] = b2bit[sym]

    start_r, start_c = valley.start
    goal_r, goal_c = valley.goal
    period = valley.tide_period
    low = valley.tide_low

    dist = {}
    parent = {}

    init_state = (start_r, start_c, init_mask, 0)
    dist[init_state] = 0
    pq = [(0, start_r, start_c, init_mask)]

    while pq:
        cur_t, r, c, mask = heapq.heappop(pq)
        cur_mod = cur_t % period

        if cur_t > dist.get((r, c, mask, cur_mod), 1e9):
            continue

        if (r, c) == (goal_r, goal_c):
            actions = []
            curr = (r, c, mask, cur_mod)
            while curr in parent:
                prev_state, act = parent[curr]
                actions.append(act)
                curr = prev_state
            actions.reverse()
            return cur_t, "".join(actions)

        for act, (dr, dc) in ACTIONS.items():
            nr, nc = r + dr, c + dc
            if not valley.in_bounds(nr, nc):
                continue
            nsym = valley.cell(nr, nc)
            if nsym in (WALL, WATER):
                continue

            dt = valley.duration(r, c, nr, nc)
            nt = cur_t + dt
            nt_mod = nt % period

            if nsym == FORD and nt_mod >= low:
                continue

            if (nr, nc) in bridge_cell_mask:
                bbit = bridge_cell_mask[(nr, nc)]
                if not (mask & bbit):
                    continue

            nmask = mask
            if act != "W" and (nr, nc) in winch_mask_toggle:
                nmask ^= winch_mask_toggle[(nr, nc)]

            nstate = (nr, nc, nmask, nt_mod)
            if nt < dist.get(nstate, 1e9):
                dist[nstate] = nt
                parent[nstate] = ((r, c, mask, cur_mod), act)
                heapq.heappush(pq, (nt, nr, nc, nmask))

    return None, None


def get_map_path():
    candidates = [
        pathlib.Path("test-input.txt"),
        pathlib.Path(__file__).resolve().parent / "test-input.txt",
        pathlib.Path("train-input.txt"),
        pathlib.Path(__file__).resolve().parent / "train-input.txt",
    ]
    for p in candidates:
        if p.exists():
            return p, p.parent / "solution.txt"
    return pathlib.Path("test-input.txt"), pathlib.Path("solution.txt")


def main():
    map_path, out_path = get_map_path()
    valley = load_map(str(map_path))
    score, route = solve(valley)
    if route:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(route + "\n")


if __name__ == "__main__":
    main()
