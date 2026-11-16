# UK - Call My BLEUff: find the honest panellist of every round
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "files")
from bleu import corpus_bleu, tokenize

refs = pd.read_csv("files/references.csv")
panel = pd.read_csv("files/panel.csv")
ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()

# keyword lists: which fact value a text states
FOOD = {
    "japanese": r"japan|sushi", "italian": r"ital|pasta|pizza", "french": r"french|france",
    "english": r"english|british", "indian": r"\bindian (?!cuisine)|indian$|curry (?!\w)", "chinese": r"chinese",
    "fastfood": r"fast[- ]?food|burger|american|take ?away|fries",
}
AREA = {"riverside": r"river", "centre": r"cent(re|er)|downtown"}
PRICE = {
    "cheap": r"cheap|inexpensive|low[- ]?(price|priced|cost|pricing|end)|(less|lower) than £?20|under £?20"
             r"|budget|affordab|below £?20|bargain|prices? (is|are) low|value price|lower price",
    "moderate": r"moderate|average[- ]?pric|mid[- ]?(range|price)|medium|£?20 ?(-|to|and) ?£?25|reasonabl"
                r"|middle price|prices? (range )?(is|are) average|fairly priced|well priced",
    "high": r"expensive|high[- ]?(price|priced|cost|end)|more than £?30|pricey|costly|upscale|upmarket"
            r"|higher[- ]?(price|end|margin)|(above|over) £?30|£?30 and over|prices? (range )?(is|are) (high|more)"
            r"|highly[- ]priced|priced high|bit more",
}


def by_keywords(text, words):
    return {k for k, pattern in words.items() if re.search(pattern, text.lower())}


def rating(text):
    t = text.lower()
    found = set()
    m = re.search(r"(\d|one|three|five)\s*(out of|of|/)\s*(a )?(5|five)", t) or re.search(r"(\d|one|three|five)[- ]stars?", t)
    if m:
        found.add({"1": "low", "one": "low", "3": "avg", "three": "avg", "5": "high", "five": "high"}.get(m.group(1), "?"))
    if re.search(r"(low|poor|bad)(ly)?[- ]?(customer |star )?(rat|review)|rat\w+ (\w+ )?(low|poor)|not (well|highly) rated|underrated|don't love", t):
        found.add("low")
    if re.search(r"average (customer )?(rat|review)|rat\w+ (\w+ )?average|averagely rated", t):
        found.add("avg")
    if re.search(r"(high|great|good|top)(ly)?[- ]?(customer )?(rat|review)|rat\w+ (\w+ )?(high|excellent)|highly[- ]rated"
                 r"|well rated|excellent|stellar|high[- ]quality|customer feedback", t):
        found.add("high")
    return found


def family(text):
    t = text.lower()
    if re.search(r"adults? only|adult[- ]oriented|\b(not|non|no|isn't|aren't)\b[ -]?(\w+ ){0,3}(famil|kid|child)", t):
        return {"no"}
    return {"yes"} if re.search(r"famil|kid|child", t) else set()


def venue_names(text):
    # capitalised multi-word names, e.g. "Café Sicilia", "The Rice Boat" -> "rice boat"
    found = re.findall(r"[A-Z][\w'é]+(?: [A-Z][\w'é]+)+", text)
    return {re.sub(r"^the ", "", n.lower()) for n in found}


FACTS = {"food": lambda t: by_keywords(t, FOOD), "area": lambda t: by_keywords(t, AREA),
         "price": lambda t: by_keywords(t, PRICE), "rating": rating, "family": family, "name": venue_names}


def missing_facts(text, references):
    """Facts that at least 3 of the 5 references state but the text does not."""
    missing = []
    for fact, extract in FACTS.items():
        counts = Counter(v for ref in references for v in extract(ref))
        agreed = {v for v, c in counts.items() if c >= 3}
        if fact == "name":
            missing += [n for n in agreed if n not in text.lower()]
        elif agreed and not (extract(text) & set(counts)):
            missing.append(fact)
    return missing


def repeated_4grams(text):
    """How many 4-word sequences appear more than once in the text."""
    tok = [w for w in tokenize(text) if w.isalnum()]   # words only
    grams = Counter(tuple(tok[i:i + 4]) for i in range(len(tok) - 3))
    return sum(c - 1 for c in grams.values() if c > 1)


# check every card: bad = drops an agreed fact, or repeats itself
panel["missing"] = [missing_facts(t, ref_map[(r, c)]) for r, c, t in zip(panel["round"], panel.card, panel.text)]
panel["repeats"] = [repeated_4grams(t) for t in panel.text]
panel["bad"] = (panel.missing.str.len() > 0) | (panel.repeats > 0)

rows = []
for (rnd, seat), deck in panel.groupby(["round", "seat"]):
    bleu = corpus_bleu(list(deck.text), [ref_map[(rnd, c)] for c in deck.card])
    rows.append({"round": rnd, "seat": seat, "bleu": round(bleu["bleu"], 3),
                 "brevity_penalty": round(bleu["brevity_penalty"], 2),
                 "cards_missing_facts": int((deck.missing.str.len() > 0).sum()),
                 "cards_repeating": int((deck.repeats > 0).sum()),
                 "bad_cards": int(deck.bad.sum())})
decks = pd.DataFrame(rows)
decks.to_csv("decks.csv", index=False)
print(decks.pivot(index="round", columns="seat", values="bad_cards").to_string())


def decode(seats):
    word = ""
    for i in range(0, len(seats), 3):
        d1, d2, d3 = (s - 1 for s in seats[i:i + 3])
        word += chr(ord("A") + 9 * d1 + 3 * d2 + d3)
    return word


honest = decks.loc[decks.groupby("round").bad_cards.idxmin()].sort_values("round")
crowned = decks.loc[decks.groupby("round").bleu.idxmax()].sort_values("round")
print("honest seats:", list(honest.seat))
print("BLEU's choice: SIGMOID_" + decode(list(crowned.seat)))
print("FLAG:          SIGMOID_" + decode(list(honest.seat)))

# figure: bad cards per deck, honest deck marked
fig, ax = plt.subplots(figsize=(11, 4))
for seat, grp in decks.groupby("seat"):
    ax.scatter(grp["round"], grp.bad_cards, label=f"seat {seat}", s=30)
ax.scatter(honest["round"], honest.bad_cards, s=140, facecolors="none", edgecolors="black", label="honest (fewest bad cards)")
ax.set_xlabel("round")
ax.set_ylabel("bad cards out of 40")
ax.set_xticks(range(1, 25))
ax.legend(frameon=False, ncol=4)
fig.tight_layout()
fig.savefig("honesty_check.png", dpi=130)
