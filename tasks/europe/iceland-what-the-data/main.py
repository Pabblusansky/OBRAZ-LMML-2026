"""Clean the Icelandic station export: test-input.csv -> test-output.csv."""
import csv
import io

IN, OUT = "test-input.csv", "test-output.csv"
FIELDS = ["reading_id", "station_name", "temperature_c", "reading_date"]


def fix_name(s):
    # names are UTF-8 read as latin-1/cp1252: "MÃ½vatn" -> "Mývatn".
    # only touched when the mojibake markers are still there, so good names survive.
    s = s.strip()
    for _ in range(2):
        if "Ã" not in s and "Â" not in s:
            break
        for enc in ("latin-1", "cp1252"):
            try:
                t = s.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if t != s:
                s = t
            break
        else:
            break
    return s


def main():
    data = open(IN, "rb").read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            pass

    lines = text.splitlines()
    # preamble length varies between exports: find the header, never count lines
    start = next(i for i, l in enumerate(lines) if "reading_id" in l and ";" in l)

    rows = {}
    for rec in csv.reader(io.StringIO("\n".join(lines[start + 1:])), delimiter=";"):
        if len(rec) < 5 or not rec[0].strip().isdigit():
            continue
        rid, name, temp, unit, date = (f.strip() for f in rec[:5])
        row = (rid, fix_name(name), float(temp.replace(",", ".")), unit.upper()[:1], date)
        prev = rows.get(rid)
        # duplicates are retransmissions; a Celsius copy wins so no round-off is added
        if prev is None or (prev[3] != "C" and unit.upper()[:1] == "C"):
            rows[rid] = row

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for rid in sorted(rows, key=int):
            _, name, t, unit, date = rows[rid]
            t = t if unit == "C" else (t - 32) * 5 / 9
            w.writerow([rid, name, f"{t:.1f}", date])


if __name__ == "__main__":
    main()
