# Costa Rica - "Pura Vida, No PII": anonimizarea mesajelor clientilor agentiei Pura Vida Voyages.
#
# Ce face: primeste mesaje (e-mailuri, chat-uri) si inlocuieste fiecare data personala
# cu o eticheta: [EMAIL], [ACCOUNT] (IBAN), [DOB] (data nasterii), [PHONE], [ID] (cedula),
# [ADDRESS], [NAME]. Referintele de rezervare PV-xxxxx si destinatiile turistice raman.
#
# Cum: un sir de etape bazate pe expresii regulate
# Doua idei tin totul impreuna:
#   - "Seiful" (Vault): ce a fost deja inlocuit e ascuns in spatele unor caractere speciale,
#     ca etapele urmatoare sa nu-l mai poata atinge.
#   - "Trecerile de variante": dupa ce o valoare e gasita o data (un telefon, o data, un
#     nume), o cautam si in celelalte forme in care poate fi scrisa in acelasi mesaj.

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Callable

# Marcaje ("santinele") din zona Unicode de uz privat (U+E000...U+F8FF).
# Aceste caractere nu apar in text normal si nu sunt nici litere, nici cifre, deci nicio
# expresie regulata din etapele urmatoare (\w, \d, [A-Z]...) nu le poate potrivi.
_SENTINEL_OPEN = "\uE000"
_SENTINEL_CLOSE = "\uE001"
# Tabel de traducere: cifra 0..9 -> caracterul privat U+E010..U+E019.
# Indexul din seif nu poate fi scris cu cifre obisnuite, altfel regulile pentru telefoane
# sau cedulas l-ar putea lua drept numar si l-ar inlocui.
_DIGIT_TO_PUA = str.maketrans("0123456789", "\uE010\uE011\uE012\uE013\uE014\uE015\uE016\uE017\uE018\uE019")


# Transforma indexul (ex. 12) in sirul privat corespunzator (U+E011 U+E012).
def _encode_index(idx: int) -> str:
    return str(idx).translate(_DIGIT_TO_PUA)


# "Seiful". Problema: ce am inlocuit deja (ex. "[ACCOUNT]") sau ce trebuie pastrat
# (ex. referinta "PV-AB123", care contine cifre) nu trebuie atins de etapele urmatoare.
# Solutia: mutam valoarea intr-o lista si in text punem doar un marcaj
# (U+E000 + index codat + U+E001). La final, restore() pune valorile inapoi.
class Vault:
    """Seif pentru bucatile protejate si placeholderele emise, ascunse in spatele santinelelor."""

    def __init__(self) -> None:
        self._items: list[str] = []

    # Salveaza valoarea si intoarce marcajul care o inlocuieste in text.
    def stash(self, value: str) -> str:
        # Pozitia noii valori in lista devine indexul ei
        idx = len(self._items)
        self._items.append(value)
        return f"{_SENTINEL_OPEN}{_encode_index(idx)}{_SENTINEL_CLOSE}"

    # Inlocuieste fiecare marcaj din text cu valoarea salvata.
    def restore(self, text: str) -> str:
        # Dictionarul invers: caracter privat -> cifra obisnuita
        pua_to_digit = {chr(0xE010 + digit): str(digit) for digit in range(10)}
        # Expresia unui marcaj intreg: deschidere, una sau mai multe cifre private, inchidere
        pattern = re.compile(
            re.escape(_SENTINEL_OPEN)
            + r"([\uE010-\uE019]+)"
            + re.escape(_SENTINEL_CLOSE)
        )

        # Decodam indexul si intoarcem valoarea originala din seif
        def _repl(match: re.Match[str]) -> str:
            digits = "".join(pua_to_digit[char] for char in match.group(1))
            return self._items[int(digits)]

        # Doua treceri: daca o valoare salvata continea la randul ei un marcaj, a doua
        # trecere il rezolva si pe acela.
        restored = pattern.sub(_repl, text)
        restored = pattern.sub(_repl, restored)
        return restored


# ===== Normalizare: comparatii fara majuscule si fara accente =====
# NFD descompune fiecare litera accentuata in litera de baza + semnul diacritic
# ("é" -> "e" + U+0301). Semnele diacritice au categoria Unicode "Mn"; le eliminam si
# trecem la litere mici: "José" -> "jose", "Río" -> "rio".
def strip_accents_lower(input_text: str) -> str:
    decomposed = unicodedata.normalize("NFD", input_text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").lower()


# Problema: cautam tiparele pe textul normalizat (fara accente, litere mici), dar trebuie
# sa inlocuim in textul ORIGINAL. De aceea construim, pe langa textul normalizat, lista
# "spans": spans[i] = (start, end) = pozitia in original a caracterului i normalizat.
def build_normalized_index(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Intoarce (text_fara_accente_cu_litere_mici, [(start_original, end_original), ...])."""
    norm_chars: list[str] = []
    spans: list[tuple[int, int]] = []
    for char_idx, char in enumerate(text):
        char_category = unicodedata.category(char)
        # Semn diacritic separat (combinant): nu produce caracter normalizat, dar il
        # lipim de litera dinaintea lui, extinzand capatul intervalului ei.
        if char_category == "Mn":
            if spans:
                span_start, _ = spans[-1]
                spans[-1] = (span_start, char_idx + 1)
            continue
        # Caracter obisnuit: il descompunem si pastram doar literele de baza
        decomp = unicodedata.normalize("NFD", char)
        base_chars = [sub_char.lower() for sub_char in decomp if unicodedata.category(sub_char) != "Mn"]
        # Nu a ramas nimic (caracter format doar din diacritice) -> il sarim
        if not base_chars:
            continue
        # Fiecare litera de baza indica spre acelasi caracter original char_idx
        for base_char in base_chars:
            norm_chars.append(base_char)
            spans.append((char_idx, char_idx + 1))
    return "".join(norm_chars), spans


# Aplica mai multe inlocuiri (start, end, text_nou) deodata, intr-o singura reconstructie.
def replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Aplica pe text editari (start, end, inlocuire) care nu se suprapun."""
    if not spans:
        return text
    # Sortam dupa start; la acelasi start punem intai intervalul cel mai lung,
    # ca potrivirea mai completa sa castige.
    spans_sorted = sorted(spans, key=lambda span: (span[0], -(span[1] - span[0])))
    # Pastram doar intervalele care nu se suprapun cu unul deja acceptat
    merged: list[tuple[int, int, str]] = []
    cur_end = -1
    for start, end, repl in spans_sorted:
        # Incepe inainte sa se termine cel precedent -> suprapunere, il ignoram
        if start < cur_end:
            continue
        merged.append((start, end, repl))
        cur_end = end
    # Reconstruim textul: bucata neatinsa dintre editari, apoi inlocuirea, pe rand
    parts: list[str] = []
    cursor = 0
    for start, end, repl in merged:
        parts.append(text[cursor:start])
        parts.append(repl)
        cursor = end
    # Coada textului de dupa ultima editare
    parts.append(text[cursor:])
    return "".join(parts)


# Echivalentul lui re.sub, dar insensibil la majuscule si accente: cautam tiparul pe
# textul normalizat si inlocuim in textul original, folosind harta de pozitii.
# repl_fn primeste bucata originala si potrivirea, si intoarce textul de inlocuire.
def sub_normalized(
    text: str,
    pattern: re.Pattern[str],
    repl_fn: Callable[[str, re.Match[str]], str],
) -> str:
    """Cauta tiparul pe textul normalizat si inlocuieste in textul original."""
    norm_text, spans = build_normalized_index(text)
    edits: list[tuple[int, int, str]] = []
    for regex_match in pattern.finditer(norm_text):
        # Ignoram potrivirile goale (lungime zero)
        if regex_match.start() == regex_match.end():
            continue
        # Traducem pozitiile din textul normalizat in pozitii din textul original
        orig_start = spans[regex_match.start()][0]
        orig_end = spans[regex_match.end() - 1][1]
        orig_slice = text[orig_start:orig_end]
        replacement = repl_fn(orig_slice, regex_match)
        # Inregistram editarea doar daca s-a schimbat ceva
        if replacement != orig_slice:
            edits.append((orig_start, orig_end, replacement))
    return replace_spans(text, edits)


# ===== Lexicoane si constante =====
# Numele lunilor in engleza si spaniola (cu abrevieri) -> numarul lunii.
# Cheile sunt fara accente si cu litere mici, ca textul normalizat.
MONTH_NAME_TO_NUM: dict[str, int] = {
    # Engleza
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # Spaniola
    "enero": 1, "ene": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4, "abr": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "setiembre": 9, "set": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12, "dic": 12,
}

# Inversul: numarul lunii -> toate denumirile ei. Il folosim cand am gasit o data si vrem
# sa stergem si celelalte scrieri ale ei (ex. "26/03/1975" si "marzo 26, 1975").
NUM_TO_MONTHS: dict[int, tuple[str, ...]] = {
    1: ("january", "jan", "enero", "ene"),
    2: ("february", "feb", "febrero"),
    3: ("march", "mar", "marzo"),
    4: ("april", "apr", "abril", "abr"),
    5: ("may", "mayo"),
    6: ("june", "jun", "junio"),
    7: ("july", "jul", "julio"),
    8: ("august", "aug", "agosto", "ago"),
    9: ("september", "sep", "sept", "septiembre", "setiembre", "set"),
    10: ("october", "oct", "octubre"),
    11: ("november", "nov", "noviembre"),
    12: ("december", "dec", "diciembre", "dic"),
}

# Alternativa regex cu toate lunile ("september|...|sep|..."). Cele mai lungi intai,
# altfel "sep" ar castiga in fata lui "september" si ar ramane "tember" nepotrivit.
_MONTH_ALT = "|".join(sorted(MONTH_NAME_TO_NUM.keys(), key=len, reverse=True))

# Fraze cu majuscule care NU sunt date personale: numele agentiei si destinatii turistice.
# Fara aceasta lista, "Manuel Antonio" (parc national) ar fi luat drept prenume + nume de
# persoana. Inainte de etapa de nume, aceste fraze sunt mutate in seif.
PROTECTED_NON_PII_PHRASES: tuple[str, ...] = (
    "Pura Vida Voyages",
    "Pura Vida",
    "Costa Rica",
    "San José",
    "San Jose",
    "Manuel Antonio",
    "Rincón de la Vieja",
    "Rincon de la Vieja",
    "Puerto Viejo",
    "La Fortuna",
    "Santa Teresa",
    "Playa Conchal",
    "Playa Flamingo",
    "Playa Hermosa",
    "Playa del Coco",
    "Playas del Coco",
    "Bahía Drake",
    "Bahia Drake",
    "Drake Bay",
    "Río Celeste",
    "Rio Celeste",
    "Braulio Carrillo",
    "San Gerardo de Dota",
    "Barra del Colorado",
    "San Carlos",
    "Santa Elena",
    "Santa Ana",
    "Santa Rosa",
    "Santo Domingo",
    "San Pedro",
    "San Isidro",
    "San Ramón",
    "San Ramon",
    "San Rafael",
    "San Francisco",
    "San Gabriel",
    "San Pablo",
    "San Antonio",
    "San Mateo",
    "San Marcos",
    "San Lucas",
    "San Juan",
    "San Miguel",
    "San Sebastián",
    "San Sebastian",
    "San Joaquín",
    "San Joaquin",
    "San Vicente",
    "Santa Cruz",
    "Santa Bárbara",
    "Santa Barbara",
    "Santa Marta",
    "Santa Clara",
    "Santa María",
    "Santa Maria",
    "Santa Lucía",
    "Santa Lucia",
    "Arenal",
    "Tortuguero",
    "Tamarindo",
    "Monteverde",
    "Corcovado",
    "Guanacaste",
    "Montezuma",
    "Nosara",
    "Sámara",
    "Samara",
    "Jacó",
    "Jaco",
    "Dominical",
    "Uvita",
    "Chirripó",
    "Chirripo",
    "Sarapiquí",
    "Sarapiqui",
    "Cahuita",
    "Poás",
    "Poas",
    "Irazú",
    "Irazu",
    "Turrialba",
    "Papagayo",
    "Nicoya",
    "Pacuare",
    "Mal País",
    "Mal Pais",
    "Osa Peninsula",
)

# Orase si provincii din Costa Rica. Apar la finalul unei adrese ("..., Escazu"), dar
# singure nu identifica pe nimeni si apar des in text turistic normal. De aceea NU le
# stergem separat in restul mesajului, ci doar ca parte a adresei complete.
ADDRESS_CITY_NAMES: frozenset[str] = frozenset({
    "san jose", "quepos", "heredia", "limon", "escazu", "puntarenas",
    "alajuela", "cartago", "liberia", "santa ana", "curridabat", "san pedro",
    "desamparados", "moravia", "tibas", "montes de oca", "rohrmoser", "pavas",
    "jaco", "tamarindo", "san isidro", "san carlos", "perez zeledon", "grecia",
    "turrialba", "belen", "santo domingo", "san rafael", "san francisco",
    "guadalupe", "goicoechea", "tres rios", "la union", "san ramon", "ciudad quesada",
    "nicoya", "santa cruz", "cañas", "canas", "golfito", "ciudad neily",
    "guapiles", "siquirres", "paraiso", "atentas", "atenas", "naranjo", "palmares",
    "orosi", "barva", "flores", "sarchi", "santa barbara", "san pablo", "la fortuna",
    "manuel antonio", "monteverde", "puerto viejo", "nosara", "samara", "uvita",
    "dominical", "costa rica",
})

# Cuvinte care nu sunt nume, chiar daca apar cu majuscula la inceput de fraza sau ca
# parte locala a unui e-mail (info@, booking@...). Previn stergerile gresite.
NON_NAME_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "have", "has", "had",
    "are", "was", "were", "will", "would", "could", "should", "can", "may",
    "not", "but", "our", "your", "their", "his", "her", "its", "you", "they",
    "she", "him", "them", "who", "what", "when", "where", "why", "how", "all",
    "any", "each", "some", "more", "most", "other", "such", "only", "also",
    "than", "then", "now", "here", "there", "very", "just", "well", "too",
    "info", "contact", "admin", "support", "sales", "booking", "bookings",
    "travel", "travels", "trip", "trips", "tour", "tours", "mail", "email",
    "correo", "user", "client", "customer", "office", "noreply", "billing",
    "refund", "refunds", "payment", "payments", "team", "hola", "hello", "dear",
    "hi", "regards", "sincerely", "thanks", "thank", "best", "cheers",
    "urgent", "priority", "rush", "hurry", "please", "confirm", "update",
    "updated", "itinerary", "invoice", "deposit", "balance", "account",
    "bank", "wire", "transfer", "charge", "double", "complaint", "service",
    "guide", "hotel", "stay", "flight", "reschedule", "postpone", "cancel",
    "cancelled", "canceled", "date", "dates", "day", "days", "week", "month",
    "morning", "afternoon", "evening", "number", "phone", "call", "callback",
    "identity", "verify", "cedula", "address", "pickup", "paperwork", "reply",
    "written", "details", "passenger", "list", "family", "son", "daughter",
    "wife", "husband", "partner", "spouse", "brother", "sister", "mother",
    "father", "friend", "name", "attached", "file", "reference", "keep",
    "send", "add", "check", "move", "change", "want", "need", "like", "make",
    "help", "let", "ask", "asking", "waiting", "received", "settled", "agreed",
    "fine", "good", "bad", "poor", "late", "early", "upset", "disappointed",
    "unacceptable", "immediately", "soon", "possible", "convenience",
    "puravida", "pura", "vida", "voyages", "costa", "rica", "san", "jose",
    "manuel", "antonio", "arenal", "tortuguero", "tamarindo", "quepos",
    "heredia", "limon", "escazu", "puntarenas",
})

# Lexicon de prenume (costaricane, hispanice, englezesti), fara accente, litere mici.
FIRST_NAMES: frozenset[str] = frozenset({
    "adriana", "agustin", "aaron", "alberto", "alejandra", "alejandro", "alex",
    "alexa", "alexander", "alexandra", "alfonso", "alfredo", "alicia", "alonso",
    "alvaro", "amanda", "amelia", "ana", "anahi", "andrea", "andres", "angel",
    "angela", "angelica", "anthony", "antonia", "antonio", "araceli", "ariana",
    "ariel", "armando", "arnoldo", "arturo", "astrid", "augusto", "aurora",
    "axel", "barbara", "beatriz", "belen", "benjamin", "bernardo", "berta",
    "bianca", "blanca", "braulio", "brenda", "brian", "bruno", "bryan",
    "camila", "camilo", "carla", "carlos", "carmela", "carmen", "carolina",
    "catalina", "cecilia", "celeste", "cesar", "christian", "christopher",
    "cindy", "clara", "claudia", "clemente", "concepcion", "consuelo", "cristian",
    "cristina", "cristobal", "cynthia", "damian", "daniel", "daniela", "dante",
    "dario", "david", "dayana", "debora", "deborah", "diana", "diego", "dolores",
    "domingo", "doris", "dylan", "edgar", "edgardo", "edith", "eduardo",
    "edwin", "efrain", "elena", "elias", "elisa", "elisabeth", "elizabeth",
    "eloisa", "elsa", "elvira", "emanuel", "emilia", "emiliano", "emilio",
    "emily", "emma", "enrique", "eric", "erica", "erick", "erik", "erika",
    "ernesto", "esmeralda", "esperanza", "esteban", "estefania", "estela",
    "ester", "esther", "estrella", "eugenia", "eugenio", "eva", "evelyn",
    "fabian", "fabiana", "fabiola", "fatima", "federico", "felipe", "felix",
    "fernanda", "fernando", "fidel", "fiorella", "flavia", "flor", "florencia",
    "francisca", "francisco", "franco", "freddy", "gabriel", "gabriela", "gael",
    "gema", "genesis", "georgina", "gerardo", "german", "gilberto", "gina",
    "giovanna", "giovanni", "gisela", "giselle", "gladis", "gladys", "gloria",
    "gonzalo", "graciela", "gregorio", "grettel", "guadalupe", "guillermo",
    "gustavo", "hannah", "hector", "helena", "henry", "heriberto", "hernan",
    "hilda", "hugo", "humberto", "ian", "ignacio", "iker", "ileana", "iliana",
    "ines", "ingrid", "irene", "iris", "irma", "isabel", "isabela", "isabella",
    "isaias", "isidro", "ismael", "israel", "ivan", "ivana", "iveth", "ivette",
    "ivonne", "jacinto", "jacob", "jacqueline", "jaime", "jairo", "james",
    "janet", "janeth", "jason", "javier", "jean", "jeannette", "jennifer",
    "jenny", "jeremy", "jeronimo", "jessica", "jesus", "jimena", "joan",
    "joaquim", "joaquin", "joel", "johan", "johanna", "john", "jonas",
    "jonathan", "jordi", "jorge", "jose", "josefa", "josefina", "joselyn",
    "joshua", "josue", "juan", "juana", "juanita", "judith", "julia", "julian",
    "juliana", "julio", "karen", "karina", "karla", "katerin", "katherine",
    "katia", "katty", "keilyn", "kelly", "kenneth", "kevin", "keylor", "kimberly",
    "laura", "lazaro", "leandro", "leon", "leonardo", "leonor", "leticia",
    "leyla", "lia", "lidia", "ligia", "liliana", "lillian", "lina", "linda",
    "lisbeth", "lissette", "lizbeth", "lizeth", "lorena", "lorenzo", "lourdes",
    "lucas", "lucia", "luciana", "luciano", "lucio", "lucrecia", "luis",
    "luisa", "luz", "lydia", "magdalena", "maikel", "maite", "manfred",
    "manuel", "manuela", "marcela", "marcelo", "marcia", "marco", "marcos",
    "margarita", "maria", "mariana", "marianela", "mariano", "maribel",
    "maricela", "mariela", "marielos", "marilin", "marina", "mario", "marisol",
    "maritza", "marlene", "marlon", "marta", "martha", "martin", "mateo",
    "matias", "matilde", "mauricio", "mauro", "maximiliano", "maximo", "mayela",
    "mayra", "melanie", "melany", "melina", "melissa", "mercedes", "michael",
    "michel", "michelle", "miguel", "milagro", "milagros", "milena", "miriam",
    "mirna", "monica", "monserrat", "montserrat", "nadia", "nancy", "naomi",
    "natalia", "natalie", "nathalia", "nathalie", "nayeli", "nelson", "nestor",
    "nicolas", "nicole", "nidia", "noelia", "noemi", "nora", "norma", "nuria",
    "octavio", "odette", "ofelia", "olga", "oliver", "olivia", "omar", "orlando",
    "oscar", "osvaldo", "otto", "pablo", "paloma", "pamela", "paola", "paolo",
    "patricia", "patricio", "patrick", "paula", "paulina", "pedro", "perla",
    "pilar", "priscila", "rafael", "raissa", "ramiro", "ramon", "randall",
    "raquel", "raul", "rebecca", "rebeca", "regina", "reinaldo", "renata",
    "renato", "rene", "reyes", "reynaldo", "ricardo", "richard", "rita",
    "roberto", "rocio", "rodolfo", "rodrigo", "rogelio", "rolando", "roman",
    "romina", "ronald", "ronny", "rosa", "rosalia", "rosalinda", "rosario",
    "rosita", "rossana", "roxana", "roy", "ruben", "ruth", "sabrina", "salome",
    "salvador", "samantha", "samuel", "sandra", "santiago", "sara", "sarah",
    "saul", "sebastian", "selena", "sergio", "sharon", "sheila", "shirley",
    "silvana", "silvia", "simon", "sofia", "sol", "solange", "soledad", "sonia",
    "sophia", "soraya", "stefania", "stefany", "stephanie", "steven", "susana",
    "tamara", "tania", "tatiana", "teresa", "thiago", "thomas", "tomas",
    "valentin", "valentina", "valeria", "valerie", "vanesa", "vanessa",
    "veronica", "vicente", "victor", "victoria", "vilma", "vinicio", "violeta",
    "virginia", "vivian", "viviana", "walter", "wendy", "wilberth", "wilfredo",
    "william", "wilmer", "ximena", "xiomara", "yadira", "yahaira", "yamileth",
    "yaneth", "yendry", "yesenia", "yessenia", "yolanda", "yuliana", "yuri",
    "zaida", "zoe", "zulema",
})

# Lexicon de nume de familie, fara accente, litere mici. In etapa 8c o fraza e
# recunoscuta ca nume daca incepe cu un prenume din FIRST_NAMES si contine cel putin un
# nume de familie din LAST_NAMES.
LAST_NAMES: frozenset[str] = frozenset({
    "abarca", "acevedo", "acosta", "acuna", "aguero", "aguilar", "aguilera",
    "aguirre", "alanis", "aleman", "alfaro", "allen", "almanza", "alonso",
    "alpizar", "altamirano", "alvarado", "alvarez", "amador", "anaya", "anderson",
    "andrade", "angulo", "aragon", "arana", "araujo", "araya", "arce",
    "arellano", "arenas", "arguedas", "arias", "armas", "armijo", "arredondo",
    "arrieta", "arroyo", "arteaga", "artavia", "asencio", "astorga", "astua",
    "avalos", "avendano", "avila", "aviles", "ayala", "azofeifa", "baca",
    "badilla", "baez", "baeza", "baltodano", "baquero", "barahona", "barboza",
    "barquero", "barragan", "barreda", "barrera", "barrientos", "barrios",
    "barton", "basulto", "bastos", "batista", "bautista", "bazan", "becerra",
    "bejarano", "beltran", "benavides", "benitez", "bermudez", "bernal",
    "berrios", "berrocal", "betancourt", "blanco", "blandon", "bobadilla",
    "bogantes", "bolanos", "bolivar", "bonilla", "borbon", "borge", "borges",
    "botero", "brenes", "briceno", "brito", "brown", "buendia", "buitrago",
    "burgos", "bustamante", "bustos", "caamano", "caballero", "cabanillas",
    "cabezas", "cabrera", "caceres", "cadena", "calderon", "callejas", "calvo",
    "camacho", "camargo", "cambronero", "campbell", "campos", "canales",
    "cancio", "candia", "canelas", "cano", "cantillo", "canto", "canton",
    "caraballo", "caravaca", "carbajal", "carballo", "cardenas", "cardona",
    "cardoza", "carmona", "carpio", "carranza", "carrasco", "carreno",
    "carrera", "carrillo", "carrion", "carvajal", "casares", "casas",
    "casasola", "cascante", "castaneda", "castellanos", "castillo", "castro",
    "cavazos", "ceballos", "cedeno", "celedon", "centeno", "cepeda", "cerda",
    "cerdas", "ceron", "cervantes", "cespedes", "chacon", "chamorro", "chavarria",
    "chaves", "chavez", "chinchilla", "chong", "cisneros", "clark", "cobos",
    "collado", "colindres", "conejo", "contreras", "cordero", "cordoba",
    "cornejo", "coronado", "coronel", "corral", "corrales", "correa", "cortes",
    "cortez", "coto", "crespo", "criollo", "cruz", "cuadra", "cubero",
    "cubillo", "cubillos", "cuellar", "cuenca", "cuesta", "cuevas", "davalos",
    "davila", "davis", "delgadillo", "delgado", "diaz", "dominguez", "donate",
    "donato", "duarte", "duenas", "duran", "echeverria", "elizondo", "enriquez",
    "escalante", "escobar", "eslava", "esparza", "espinal", "espino",
    "espinosa", "espinoza", "esquivel", "estevez", "estrada", "fajardo",
    "falk", "fallas", "farias", "fernandez", "ferrer", "fierro", "figueroa",
    "flores", "fonseca", "fournier", "franceschi", "franco", "frias",
    "fuentes", "gaitan", "galarza", "galicia", "galindo", "gallardo",
    "gallegos", "galvan", "galvez", "gamboa", "gamez", "gaona", "garay",
    "garbanzo", "garcia", "garita", "garnier", "garro", "garza", "garzon",
    "gatica", "gil", "giraldo", "giron", "godinez", "godoy", "gomez",
    "gonzales", "gonzalez", "gordillo", "gracia", "granados", "granja",
    "grillo", "guadamuz", "guerra", "guerrero", "guevara", "guillen",
    "gutierrez", "guzman", "hernandez", "herrera", "hidalgo", "higuera",
    "hinojosa", "huerta", "hurtado", "ibanez", "ibarra", "iglesias",
    "infante", "iniguez", "iraheta", "irizarry", "jaen", "jaime", "jaimes",
    "jara", "jaramillo", "jarquin", "jauregui", "jerez", "jimenez", "johnson",
    "jones", "jovel", "juarez", "jurado", "lara", "larrazabal", "lopez",
    "lazo", "leal", "ledesma", "leiva", "lema", "lemus", "leon", "leiva",
    "leyva", "limon", "linares", "lira", "lopez", "loredo", "lorenzo",
    "lozano", "lucero", "lugo", "lujan", "luna", "macedo", "machado",
    "macias", "madriz", "madrigal", "magana", "maldonado", "manrique",
    "manzano", "marchena", "marin", "maroto", "marquez", "marroquin", "martin",
    "martinez", "masis", "mata", "matamoros", "mateo", "matos", "matute",
    "mayorga", "mazariegos", "medina", "medrano", "mejia", "mejias", "melendez",
    "melgar", "melo", "mena", "menjivar", "mendez", "mendieta", "mendoza",
    "meneses", "mercado", "merino", "mesa", "meza", "milla", "millan",
    "miller", "miranda", "mireles", "moctezuma", "molina", "moncada", "monge",
    "montalvo", "montana", "montanez", "montano", "monteiro", "montenegro",
    "montero", "montes", "montesinos", "montiel", "montoya", "mora", "morales",
    "moran", "moreira", "moreno", "morera", "morales", "moya", "munoz",
    "murillo", "najera", "naranjo", "narvaez", "nava", "navarrete", "navarro",
    "negrete", "negron", "nieto", "nieves", "nino", "noguera", "noriega",
    "novoa", "nunez", "obando", "obregon", "ocana", "ocampo", "ochoa",
    "ojeda", "oliva", "olivares", "olivas", "oliver", "olivera", "olivo",
    "olmedo", "olmos", "olvera", "onate", "orozco", "ortega", "ortiz",
    "ortuno", "osorio", "ospina", "otero", "ovalles", "ovando", "oviedo",
    "pacheco", "padilla", "paez", "palacio", "palacios", "palma", "palomino",
    "palomo", "paniagua", "pantoja", "parada", "paramo", "pardo", "paredes",
    "parra", "parrales", "partidas", "pasos", "patino", "pava", "pavon",
    "paz", "pedraza", "pedrosa", "pedroza", "pelayo", "pena", "penaloza",
    "penaranda", "perales", "peralta", "peraza", "perdomo", "perea", "pereira",
    "perez", "picado", "pichardo", "piedra", "pina", "pineda", "pinilla",
    "pino", "pintado", "pinto", "pizarro", "plascencia", "plata", "plaza",
    "polanco", "pomares", "ponce", "porras", "portilla", "portillo", "posada",
    "posadas", "prado", "preciado", "prieto", "puente", "puentes", "puga",
    "pulido", "quesada", "quevedo", "quijada", "quijano", "quinones", "quintana",
    "quintanilla", "quintero", "quiros", "quiroz", "rabanales", "ramirez",
    "ramon", "ramos", "rangel", "rascon", "raya", "razon", "real", "rebollar",
    "recinos", "redondo", "rendon", "renteria", "resendez", "retana", "rey",
    "reyes", "reyna", "reynoso", "ribera", "ricardo", "rico", "rincon",
    "rios", "rivas", "rivera", "rivero", "riveros", "roa", "robledo",
    "robles", "rocha", "rodriguez", "rojas", "rojo", "roldan", "roman",
    "romero", "romo", "ronquillo", "roque", "rosales", "rosario", "rosas",
    "rosell", "rosero", "roura", "rubio", "rueda", "ruiz", "saavedra",
    "saballos", "saborio", "saenz", "salas", "salazar", "salcedo", "saldana",
    "saldivar", "salgado", "salguero", "salinas", "salmeron", "salomon",
    "samaniego", "sanabria", "sanchez", "sandi", "sandoval", "sanmartin",
    "santa", "santamaria", "santana", "santiago", "santillan", "santos",
    "sanz", "sarabia", "saravia", "sarmiento", "sauceda", "saucedo", "segovia",
    "segura", "sepulveda", "serna", "serrano", "serrato", "sevilla", "sierra",
    "silva", "simpson", "sisneros", "smith", "sobrado", "sojo", "solano",
    "solares", "soler", "solis", "solorzano", "somoza", "soria", "soriano",
    "sosa", "sotelo", "soto", "sotomayor", "suarez", "suazo", "tabares",
    "taboada", "talavera", "tamayo", "tapia", "taylor", "tejada", "tejeda",
    "tellez", "tello", "tenorio", "terrazas", "thomas", "tinoco", "tirado",
    "tobar", "toledo", "toribio", "toro", "torrealba", "torres", "torrez",
    "tortosa", "toscano", "tovar", "trejos", "trevino", "trinidad", "tristan",
    "trujillo", "ulloa", "umana", "urbina", "urena", "urias", "uribe",
    "urquilla", "urrea", "urrutia", "vaca", "valadez", "valdes", "valdez",
    "valdivia", "valdivieso", "valencia", "valentin", "valenzuela", "valera",
    "valerio", "valero", "valladares", "valle", "vallejo", "vallejos",
    "valles", "valverde", "varela", "vargas", "vasconcelos", "vasquez",
    "vazquez", "vega", "vela", "velasco", "velasquez", "velazquez", "velez",
    "veliz", "venegas", "ventura", "vera", "verdugo", "verduzco", "vergara",
    "viales", "viana", "vicente", "victoria", "vidal", "vides", "vieira",
    "vigil", "vila", "villa", "villalobos", "villalpando", "villalta",
    "villanueva", "villar", "villareal", "villarreal", "villarroel",
    "villasenor", "villatoro", "villegas", "villeda", "villena", "viquez",
    "vivas", "vizcaino", "volio", "walker", "white", "williams", "wilson",
    "yanez", "yepez", "zabaleta", "zacarias", "zambrana", "zambrano",
    "zamora", "zamudio", "zapata", "zaragoza", "zarate", "zavala", "zayas",
    "zelaya", "zeladon", "zepeda", "zumbado", "zuniga",
})


# ===== Validarea datelor si generarea variantelor =====
# Data trebuie sa existe in calendar: an intre 1900 si 2029, luna 1-12, ziua potrivita
# lunii (30 de zile in aprilie, iunie, septembrie, noiembrie), iar 29 februarie doar in ani
# bisecti (divizibili cu 4, dar nu cu 100, cu exceptia celor divizibili cu 400).
def _is_valid_date(year: int, month: int, day: int) -> bool:
    if not (1900 <= year <= 2029 and 1 <= month <= 12 and 1 <= day <= 31):
        return False
    if month in (4, 6, 9, 11) and day > 30:
        return False
    if month == 2:
        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        if day > (29 if is_leap else 28):
            return False
    return True


# Pentru o data gasita, generam tiparele pentru TOATE celelalte moduri de a o scrie,
# ca sa o stergem si daca apare din nou, in alt format, in acelasi mesaj.
def _dob_variants(year: int, month: int, day: int) -> list[str]:
    """Genereaza tiparele regex (pe text normalizat) pentru toate scrierile unei date."""
    year_str = f"{year:04d}"
    month_2d = f"{month:02d}"
    day_2d = f"{day:02d}"
    month_str = str(month)
    day_str = str(day)
    m_alts = "|".join(re.escape(name) for name in NUM_TO_MONTHS[month])
    # Separator permis intre zi, luna si an: spatiu, punct, slash sau cratima
    sep = r"[\s./-]"
    pats = [
        # An-luna-zi (ISO): 1975-03-26, 1975/3/26
        rf"\b{year_str}{sep}0?{month_str}{sep}0?{day_str}\b",
        # Zi/luna/an si luna/zi/an: 26/03/1975, 3-26-1975
        rf"\b0?{day_str}{sep}0?{month_str}{sep}{year_str}\b",
        rf"\b0?{month_str}{sep}0?{day_str}{sep}{year_str}\b",
        # 8 cifre lipite: 19750326, 26031975, 03261975
        rf"\b{year_str}{month_2d}{day_2d}\b",
        rf"\b{day_2d}{month_2d}{year_str}\b",
        rf"\b{month_2d}{day_2d}{year_str}\b",
        # Cu numele lunii: "March 26, 1975", "marzo 26 de 1975", "26 de marzo de 1975", "26th March 1975"
        rf"\b(?:{m_alts})\.?\s*0?{day_str}(?:st|nd|rd|th)?(?:\s*,\s*|\s+(?:de\s+)?){year_str}\b",
        rf"\b0?{day_str}(?:st|nd|rd|th)?\s*(?:de\s+)?(?:{m_alts})\.?(?:\s*(?:de|,)\s*|\s+){year_str}\b",
        # Numele lunii lipit de cifre: "march261975", "26marzo1975"
        rf"\b(?:{m_alts})0?{day_str}{year_str}\b",
        rf"\b0?{day_str}(?:{m_alts}){year_str}\b",
    ]
    return pats


# ===== Functii ajutatoare pentru adrese si nume =====
# Verifica daca textul prins dupa o fraza ca "my address is ..." chiar arata a adresa:
# - prea scurt (< 6 caractere) sau un pronume ("me", "my email") -> nu;
# - contine virgula sau cifre -> da (adresele au de obicei numere sau parti separate);
# - contine un cuvant tipic de adresa (calle, avenida, barrio, metros, frente, street...) -> da.
def _is_plausible_address_context(candidate: str) -> bool:
    norm = strip_accents_lower(candidate).strip()
    if not norm or len(norm) < 6:
        return False
    if norm in ("me", "us", "you", "them", "my email", "this email", "the email"):
        return False
    if "," in norm:
        return True
    if re.search(r"\d", norm):
        return True
    addr_kw = (
        r"\b(?:avenida|av|calle|paseo|diagonal|transversal|boulevard|blvd|"
        r"barrio|urbanizacion|residencial|condominio|ruta|carretera|casa|"
        r"metros|mts|frente|contiguo|costado|detras|street|st|avenue|ave|"
        r"road|rd|drive|dr|lane|ln|apartamento|apt|edificio)\b"
    )
    return bool(re.search(addr_kw, norm))


# Imparte o adresa gasita in fragmente distinctive (strada, cartier, reper), ca sa le
# stergem si cand reapar separat in alt loc din mesaj (ex. clientul repeta doar
# "Barrio Escalante", fara restul adresei).
def _extract_address_fragments(addr_text: str) -> list[str]:
    """Extrage fragmentele distinctive (strada, reper, casa) dintr-o adresa."""
    frags: list[str] = []
    # Partile adresei sunt separate prin virgula; curatam spatiile si punctuatia
    parts = [piece.strip(" ,.;:") for piece in addr_text.split(",") if piece.strip(" ,.;:")]
    for part_idx, part in enumerate(parts):
        norm = strip_accents_lower(part)
        # Ultima parte, daca e doar un oras cunoscut (San Jose, Tamarindo...), nu e un
        # fragment distinctiv: orasul poate aparea legitim in text turistic.
        if part_idx == len(parts) - 1 and norm in ADDRESS_CITY_NAMES and len(parts) >= 2:
            continue
        # Partile de cel putin 4 caractere devin fragmente
        if len(norm) >= 4:
            frags.append(part)
        # Adresele costaricane folosesc repere: "200 metros al norte de la Iglesia",
        # "frente al Banco Nacional". Extragem si numele reperului de dupa aceste expresii
        # de directie ("Iglesia", "Banco Nacional").
        regex_match = re.search(
            r"(?:metros\s+(?:al\s+)?(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?|"
            r"frente\s+(?:a|al|a\s+la)|contiguo\s+(?:a|al|a\s+la)|"
            r"diagonal\s+(?:a|al|a\s+la)|costado\s+(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?|"
            r"detr[aá]s\s+de(?:l|\s+la)?)\s+(?:la\s+|el\s+|los\s+|las\s+)?(.+)",
            part,
            flags=re.IGNORECASE,
        )
        if regex_match:
            inner = regex_match.group(1).strip(" ,.;:")
            inner_norm = strip_accents_lower(inner)
            # Reperul trebuie sa fie destul de lung si sa nu fie doar un oras
            if len(inner_norm) >= 5 and inner_norm not in ADDRESS_CITY_NAMES:
                frags.append(inner)
    # Eliminam duplicatele (comparand fara accente) si sortam de la cel mai lung la cel
    # mai scurt, ca un fragment lung sa fie inlocuit inaintea unei parti din el.
    unique: dict[str, str] = {}
    for frag_text in frags:
        frag_key = strip_accents_lower(frag_text)
        if frag_key and frag_key not in unique:
            unique[frag_key] = frag_text
    return sorted(unique.values(), key=len, reverse=True)


# Inregistreaza un nume gasit pentru trecerea finala (8d): numele complet normalizat
# intra in full_names, iar fiecare cuvant al lui in name_tokens (fara particule ca "de",
# "del", "la", "y"). Astfel, daca "Maria Fernanda Solis" a fost gasita o data, un "Solis"
# singur, mai jos in mesaj, va fi si el sters.
def _add_name_candidate(
    raw_name: str,
    full_names: set[str],
    name_tokens: set[str],
) -> None:
    # Spatii multiple -> unul singur; eliminam punctuatia de la capete
    cleaned = re.sub(r"\s+", " ", raw_name).strip(" ,.;:!?\"'()[]{}")
    norm_full = strip_accents_lower(cleaned)
    if not norm_full or len(norm_full) < 2:
        return
    # Un cuvant obisnuit nu e nume
    if norm_full in NON_NAME_STOPWORDS:
        return
    full_names.add(norm_full)
    # Fiecare cuvant de minim 2 litere, care nu e cuvant comun sau particula
    for token in re.split(r"[\s]+", norm_full):
        token = token.strip("'-")
        if len(token) >= 2 and token not in NON_NAME_STOPWORDS and token not in ("de", "del", "la", "las", "los", "y"):
            name_tokens.add(token)


# ===== Etapele de anonimizare =====
# Functia principala: primeste textul unui mesaj si intoarce versiunea anonimizata.
# Etapele merg de la tiparele cele mai specifice (e-mail, IBAN) la cele mai generale
# (nume). Exemplu de ce conteaza ordinea: un IBAN are 20 de cifre; daca regula de telefon
# ar rula inaintea celei de IBAN, ar taia din el o bucata de 8 cifre si ar lasa restul.
# Tot ce se inlocuieste intra in seif, ca etapele urmatoare sa nu-l mai atinga.
def redact_document(text: str) -> str:
    # Mesaj gol -> intoarcem doar [REDACTED]
    if not text or not text.strip():
        return "[REDACTED]"

    # Un seif nou pentru fiecare document
    vault = Vault()

    # Etapa 1: referintele de rezervare (PV-AB123) NU sunt date personale si trebuie
    # pastrate intacte. Contin cifre, deci regulile de telefon/ID le-ar putea strica,
    # asa ca le mutam in seif chiar de la inceput.
    pv_re = re.compile(r"\bPV-[A-Za-z0-9]{4,10}\b")
    text = pv_re.sub(lambda regex_match: vault.stash(regex_match.group(0)), text)

    # Etapa 2: adrese de e-mail -> [EMAIL].
    # Grupul 1 = partea locala (inainte de @). Lookbehind-ul si lookahead-ul se asigura
    # ca prindem e-mailul intreg, nu doar o bucata din el.
    email_re = re.compile(
        r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![A-Za-z0-9])"
    )
    # Partile locale cu separatori sau cifre (carlos.solano, ana_m88): le cautam si singure
    # in text, pentru ca sunt practic un identificator unic.
    email_locals_sep: set[str] = set()
    # Partile locale simple (carlos): le stergem singure doar la final (8e).
    email_locals_plain: set[str] = set()
    # Cuvinte de nume si nume complete gasite in document (folosite in trecerea 8d)
    doc_name_tokens: set[str] = set()
    doc_full_names: set[str] = set()

    # Pentru fiecare e-mail: retinem partea locala si cuvintele din ea, apoi il
    # inlocuim cu [EMAIL] (pus in seif).
    def _email_repl(regex_match: re.Match[str]) -> str:
        local = regex_match.group(1)
        local_norm = strip_accents_lower(local)
        if len(local_norm) >= 3 and local_norm not in NON_NAME_STOPWORDS:
            # Are separatori sau cifre -> lista "sep"; altfel lista "plain"
            if re.search(r"[._+\-\d]", local_norm):
                email_locals_sep.add(local_norm)
            else:
                email_locals_plain.add(local_norm)
        # Impartim partea locala la separatori: "carlos.solano" -> "carlos", "solano";
        # cifrele de la final sunt eliminate ("solano88" -> "solano"). Bucatile de minim
        # 3 litere, care nu sunt cuvinte comune, devin cuvinte de nume candidate.
        for part in re.split(r"[._+\-]+", local_norm):
            part_alpha = re.sub(r"\d+$", "", part)
            if (
                len(part_alpha) >= 3
                and part_alpha.isalpha()
                and part_alpha not in NON_NAME_STOPWORDS
            ):
                doc_name_tokens.add(part_alpha)
        return vault.stash("[EMAIL]")

    text = email_re.sub(_email_repl, text)

    # Partea locala aparuta singura in text (ex. "user carlos.solano") -> [EMAIL].
    # Permitem orice separator: "carlos_solano" se potriveste si cu "carlos.solano".
    for local_norm in sorted(email_locals_sep, key=len, reverse=True):
        escaped = re.escape(local_norm)
        flex = re.sub(r"\\([._+\-])", r"[._+-]", escaped)
        pat = re.compile(rf"(?<![a-z0-9._+-]){flex}(?![a-z0-9])")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[EMAIL]"))

    # Etapa 3: conturi bancare IBAN -> [ACCOUNT].
    # Retinem cifrele fiecarui IBAN gasit, ca sa-l stergem si cand reapare in alt format.
    iban_digit_cores: set[str] = set()

    # Pastram doar cifrele; daca sunt destule (>= 14), le retinem atat complet, char_category si
    # fara primele 2 (cifrele de control), pentru ca uneori clientul scrie doar numarul
    # de cont, fara "CR" si fara cifrele de control.
    def _record_iban(raw: str) -> str:
        alnum = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
        digits_only = re.sub(r"\D", "", alnum)
        if len(digits_only) >= 14:
            iban_digit_cores.add(digits_only)
            if len(digits_only) >= 18:
                iban_digit_cores.add(digits_only[2:])
        return vault.stash("[ACCOUNT]")

    # IBAN Costa Rica: "CR" + 20 de cifre, optional grupate cu spatii sau cratime
    cr_iban_re = re.compile(r"\b[Cc][Rr](?:[\s-]?\d){20}\b")
    text = cr_iban_re.sub(lambda regex_match: _record_iban(regex_match.group(0)), text)

    # IBAN general (alte tari): 2 litere mari + 2 cifre + grupuri de caractere alfanumerice
    gen_iban_re = re.compile(
        r"\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]{4}){2,7}(?:[\s-]?[A-Z0-9]{1,4})?\b"
    )

    # Il acceptam doar daca lungimea e de IBAN (15-34 caractere) si are minim 8 cifre,
    # altfel ar putea fi un cod oarecare scris cu majuscule.
    def _gen_iban_repl(regex_match: re.Match[str]) -> str:
        raw = regex_match.group(0)
        compact = re.sub(r"[^A-Z0-9]", "", raw)
        if 15 <= len(compact) <= 34 and sum(char.isdigit() for char in compact) >= 8:
            return _record_iban(raw)
        return raw

    text = gen_iban_re.sub(_gen_iban_repl, text)

    # Orice alta aparitie a cifrelor deja gasite (cu sau fara prefix de 2 litere, cu sau
    # fara separatori intre cifre) -> [ACCOUNT]
    for iban_core in sorted(iban_digit_cores, key=len, reverse=True):
        flex_digits = r"[\s-]*".join(iban_core)
        pat = re.compile(rf"(?<![a-z0-9])(?:[a-z]{{2}}[\s-]*)?{flex_digits}(?!\d)")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[ACCOUNT]"))

    # Etapa 4: date de nastere -> [DOB], in toate formatele.
    # detected_dobs retine (an, luna, zi) pentru fiecare data gasita, ca la final (4e) sa
    # stergem orice alta scriere a aceleiasi date.
    detected_dobs: set[tuple[int, int, int]] = set()

    # Inregistram data si intoarcem marcajul pentru [DOB]
    def _stash_dob(year: int, month: int, day: int) -> str:
        detected_dobs.add((year, month, day))
        return vault.stash("[DOB]")

    # 4a. Formate cu numele lunii.
    # Luna intai: "March 26, 1975", "marzo 26 de 1975", "Mar. 26th 1975"
    dob_month_first_re = re.compile(
        rf"\b(?P<mon>{_MONTH_ALT})\.?\s+(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
        rf"(?:\s*,\s*|\s+(?:de\s+)?)(?P<year>19\d{{2}}|20[0-2]\d)\b"
    )

    # Validam data inainte de a o sterge ("February 30, 1990" nu e o data reala)
    def _dob_mf_repl(orig: str, regex_match: re.Match[str]) -> str:
        dob_year = int(regex_match.group("year"))
        dob_month = MONTH_NAME_TO_NUM[regex_match.group("mon")]
        dob_day = int(regex_match.group("day"))
        if _is_valid_date(dob_year, dob_month, dob_day):
            return _stash_dob(dob_year, dob_month, dob_day)
        return orig

    text = sub_normalized(text, dob_month_first_re, _dob_mf_repl)

    # Ziua intai: "26 March 1975", "26 de marzo de 1975", "26 marzo, 1975"
    dob_day_first_re = re.compile(
        rf"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:de\s+)?"
        rf"(?P<mon>{_MONTH_ALT})\.?(?:\s*(?:de|,)\s*|\s+)(?P<year>19\d{{2}}|20[0-2]\d)\b"
    )

    # Aceeasi validare ca mai sus
    def _dob_df_repl(orig: str, regex_match: re.Match[str]) -> str:
        dob_year = int(regex_match.group("year"))
        dob_month = MONTH_NAME_TO_NUM[regex_match.group("mon")]
        dob_day = int(regex_match.group("day"))
        if _is_valid_date(dob_year, dob_month, dob_day):
            return _stash_dob(dob_year, dob_month, dob_day)
        return orig

    text = sub_normalized(text, dob_day_first_re, _dob_df_repl)

    # 4b. Format numeric ISO: an-luna-zi (1975-03-26, 1975/3/26, 1975.03.26).
    # "\2" cere ca ambii separatori sa fie acelasi caracter.
    dob_iso_re = re.compile(
        r"\b(?P<year>19\d{2}|20[0-2]\d)([./-])(?P<mon>0?[1-9]|1[0-2])\2(?P<day>0?[1-9]|[12]\d|3[01])\b"
    )

    def _dob_iso_repl(orig: str, regex_match: re.Match[str]) -> str:
        dob_year = int(regex_match.group("year"))
        dob_month = int(regex_match.group("mon"))
        dob_day = int(regex_match.group("day"))
        if _is_valid_date(dob_year, dob_month, dob_day):
            return _stash_dob(dob_year, dob_month, dob_day)
        return orig

    text = sub_normalized(text, dob_iso_re, _dob_iso_repl)

    # 4c. Numeric zi/luna/an sau luna/zi/an (cu "/", "-" sau ".").
    dob_dmy_re = re.compile(
        r"\b(?P<p1>0?[1-9]|[12]\d|3[01])([./-])(?P<p2>0?[1-9]|[12]\d|3[01])\2(?P<year>19\d{2}|20[0-2]\d)\b"
    )

    # In Costa Rica formatul standard e zi/luna/an, deci il incercam primul. Daca a doua
    # parte e > 12, nu poate fi luna, deci data e luna/zi/an (format american).
    def _dob_dmy_repl(orig: str, regex_match: re.Match[str]) -> str:
        first_num = int(regex_match.group("p1"))
        second_num = int(regex_match.group("p2"))
        dob_year = int(regex_match.group("year"))
        if second_num <= 12 and _is_valid_date(dob_year, second_num, first_num):
            # Caz ambiguu (ex. 03/04/1980): ambele interpretari sunt valide. Retinem si
            # varianta luna/zi/an, ca trecerea 4e sa stearga ambele scrieri posibile.
            if first_num <= 12 and first_num != second_num and _is_valid_date(dob_year, first_num, second_num):
                detected_dobs.add((dob_year, first_num, second_num))
            return _stash_dob(dob_year, second_num, first_num)
        # Nu era zi/luna/an valid -> incercam luna/zi/an
        if first_num <= 12 and _is_valid_date(dob_year, first_num, second_num):
            return _stash_dob(dob_year, first_num, second_num)
        return orig

    text = sub_normalized(text, dob_dmy_re, _dob_dmy_repl)

    # 4d. 8 cifre lipite, dar DOAR dupa o expresie de data nasterii ("date of birth is
    # 19750326", "nacimiento: 26031975"). Fara context, 8 cifre ar putea fi un telefon.
    dob_glued_ctx_re = re.compile(
        r"\b(?:date of birth|dob|born|nacimiento)\s*(?:is\s+|:?\s*)(?P<digits>\d{8})\b"
    )

    def _dob_glued_repl(orig: str, regex_match: re.Match[str]) -> str:
        glued_digits = regex_match.group("digits")
        # Pastram expresia de context ("date of birth is ") si inlocuim doar cifrele
        prefix = orig[:-len(glued_digits)]
        # Incercam pe rand: AAAALLZZ, apoi ZZLLAAAA, apoi LLZZAAAA; prima valida castiga
        ymd_year, ymd_month, ymd_day = int(glued_digits[:4]), int(glued_digits[4:6]), int(glued_digits[6:8])
        if _is_valid_date(ymd_year, ymd_month, ymd_day):
            return prefix + _stash_dob(ymd_year, ymd_month, ymd_day)
        dmy_day, dmy_month, dmy_year = int(glued_digits[:2]), int(glued_digits[2:4]), int(glued_digits[4:8])
        if _is_valid_date(dmy_year, dmy_month, dmy_day):
            return prefix + _stash_dob(dmy_year, dmy_month, dmy_day)
        mdy_month, mdy_day, mdy_year = int(glued_digits[:2]), int(glued_digits[2:4]), int(glued_digits[4:8])
        if _is_valid_date(mdy_year, mdy_month, mdy_day):
            return prefix + _stash_dob(mdy_year, mdy_month, mdy_day)
        return orig

    text = sub_normalized(text, dob_glued_ctx_re, _dob_glued_repl)

    # 4e. Trecerea de variante: odata gasita o data, stergem orice alta scriere a ei
    # din mesaj (lista de tipare vine din _dob_variants).
    for dob_year, dob_month, dob_day in sorted(detected_dobs):
        for pat_str in _dob_variants(dob_year, dob_month, dob_day):
            text = sub_normalized(
                text,
                re.compile(pat_str),
                lambda _orig, _match: vault.stash("[DOB]"),
            )

    # Etapa 5: numere de telefon -> [PHONE].
    # Ordinea conteaza: intai cele internationale (cu prefix), apoi cedulas (etapa 6, 9 cifre)
    # si abia apoi telefoanele nationale de 8 cifre (5c). Altfel o cedula de 9 cifre ar fi
    # taiata intr-un "telefon" de 8 cifre + o cifra ramasa in text.
    # detected_phone_cores retine cifrele fiecarui telefon gasit, pentru trecerea de variante.
    detected_phone_cores: set[str] = set()

    # Pastram doar cifrele. Pentru +506 XXXX XXXX retinem cele 8 cifre fara prefix,
    # ca sa recunoastem acelasi numar si cand e scris fara prefix.
    def _record_phone(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 11 and digits.startswith("506"):
            detected_phone_cores.add(digits[3:])
        elif len(digits) == 8:
            detected_phone_cores.add(digits)
        elif len(digits) >= 7:
            detected_phone_cores.add(digits)
        return vault.stash("[PHONE]")

    # 5a. Telefon Costa Rica cu prefix: +506, 00506 sau 506 (eventual intre paranteze),
    # urmat de 8 cifre cu prima cifra intre 2 si 8 (telefoane fixe si mobile)
    cr_intl_phone_re = re.compile(
        r"(?<![A-Za-z0-9])(?:\+|00)?\s*\(?\s*506\s*\)?[\s.-]*[2-8]\d{3}[\s.-]*\d{4}\b"
    )
    text = cr_intl_phone_re.sub(lambda regex_match: _record_phone(regex_match.group(0)), text)

    # 5b. Alte telefoane internationale: "+" + cod de tara (1-3 cifre), optional prefix
    # intre paranteze, apoi grupuri de cifre (ex. +1 555-123-4567, +34 612 345 678)
    gen_intl_phone_re = re.compile(
        r"(?<![A-Za-z0-9])\+\d{1,3}[\s.-]*(?:\(\d{1,4}\)[\s.-]*)?\d{2,4}(?:[\s.-]*\d{2,4}){1,3}\b"
    )

    # Acceptam doar numere cu 8-15 cifre (15 = lungimea maxima a unui numar international)
    def _gen_intl_phone_repl(regex_match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", regex_match.group(0))
        if 8 <= len(digits) <= 15:
            return _record_phone(regex_match.group(0))
        return regex_match.group(0)

    text = gen_intl_phone_re.sub(_gen_intl_phone_repl, text)

    # Etapa 6: numere de identitate (cedulas) -> [ID]
    detected_cedulas: set[str] = set()

    # Retinem cifrele cedulei, pentru trecerea de variante de mai jos
    def _record_cedula(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        if digits:
            detected_cedulas.add(digits)
        return vault.stash("[ID]")

    # Cedula de persoana fizica: 9 cifre scrise X-XXXX-XXXX sau X XXXX XXXX
    # (prima cifra, 1-9, este provincia)
    cedula_sep_re = re.compile(r"\b[1-9][\s-]\d{4}[\s-]\d{4}\b")
    text = cedula_sep_re.sub(lambda regex_match: _record_cedula(regex_match.group(0)), text)

    # Cedula juridica (a unei firme): X-XXX-XXXXXX
    cedula_jur_re = re.compile(r"\b[1-9]-\d{3}-\d{6}\b")
    text = cedula_jur_re.sub(lambda regex_match: _record_cedula(regex_match.group(0)), text)

    # 9-12 cifre lipite dupa un cuvant de identitate ("cedula", "identity", "DIMEX", "ID").
    # DIMEX-ul (actul strainilor rezidenti) are 11-12 cifre. Pastram cuvantul de context
    # si inlocuim doar cifrele.
    cedula_ctx_re = re.compile(
        r"\b(?:cedula|identidad|identity|dimex|id)\s*(?:number\s+is\s+|is\s+|#?\s*:?\s*)(?P<ced>\d{9,12})\b"
    )
    text = sub_normalized(
        text,
        cedula_ctx_re,
        lambda orig, regex_match: orig[:-len(regex_match.group("ced"))] + _record_cedula(regex_match.group("ced")),
    )

    # Cedula lipita (9 cifre) oriunde in text. Lookbehind-ul exclude sumele de bani
    # ($, ₡ - colonul costarican, €, £), iar lookahead-ul exclude procentele.
    cedula_glued_re = re.compile(r"(?<![$₡€£\d])\b[1-9]\d{8}\b(?![\d%])")
    text = cedula_glued_re.sub(lambda regex_match: _record_cedula(regex_match.group(0)), text)

    # Trecerea de variante: aceeasi cedula scrisa altfel
    # (1-2345-6789 <-> 123456789 <-> 1 2345 6789) -> [ID]
    for cedula_digits in sorted(detected_cedulas, key=len, reverse=True):
        if len(cedula_digits) == 9:
            ced_province, ced_middle, ced_last = cedula_digits[0], cedula_digits[1:5], cedula_digits[5:9]
            pat = re.compile(rf"\b{ced_province}[\s.-]*{ced_middle}[\s.-]*{ced_last}\b")
        else:
            flex = r"[\s.-]*".join(cedula_digits)
            pat = re.compile(rf"\b{flex}\b")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[ID]"))

    # 5c. Telefoane nationale de 8 cifre (dupa cedulas, vezi explicatia de la etapa 5).
    # Cu separator: 8812-3456, 2222 3333. Sumele de bani si procentele sunt excluse.
    cr_nat_sep_phone_re = re.compile(r"(?<![$₡€£\d])\b[2-8]\d{3}[\s.-]\d{4}\b(?![\d%])")
    text = cr_nat_sep_phone_re.sub(lambda regex_match: _record_phone(regex_match.group(0)), text)

    # Format american, cu prefixul de zona intre paranteze: (555) 123-4567
    us_phone_re = re.compile(r"(?<![A-Za-z0-9])\(\d{3}\)[\s.-]*\d{3}[\s.-]*\d{4}\b")
    text = us_phone_re.sub(lambda regex_match: _record_phone(regex_match.group(0)), text)

    # 8 cifre lipite, prima intre 2 si 8: 88123456
    cr_nat_glued_phone_re = re.compile(r"(?<![$₡€£\d])\b[2-8]\d{7}\b(?![\d%])")
    text = cr_nat_glued_phone_re.sub(lambda regex_match: _record_phone(regex_match.group(0)), text)

    # Trecerea de variante: acelasi numar scris altfel (cu sau fara +506 / 00506,
    # cu alte separatoare sau fara) -> [PHONE]
    for phone_core in sorted(detected_phone_cores, key=len, reverse=True):
        if len(phone_core) == 8:
            phone_head, phone_tail = phone_core[:4], phone_core[4:]
            pat = re.compile(
                rf"(?<![a-z0-9])(?:\+?\s*0?0?\s*506[\s.-]*)?{phone_head}[\s.-]*{phone_tail}(?!\d)"
            )
        else:
            flex = r"[\s.-]*".join(phone_core)
            pat = re.compile(rf"(?<![a-z0-9])\+?\s*{flex}(?!\d)")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[PHONE]"))

    # Etapa 7: adrese postale -> [ADDRESS], plus fragmentele lor distinctive oriunde reapar
    detected_addr_fragments: list[str] = []

    # Retinem fragmentele adresei si intoarcem marcajul pentru [ADDRESS]
    def _record_address(raw: str) -> str:
        detected_addr_fragments.extend(_extract_address_fragments(raw))
        return vault.stash("[ADDRESS]")

    # 7a. Adresa costaricana in 2-3 parti separate prin virgula, care incepe cu un cuvant de
    # strada sau cartier: "Calle 5, Barrio Amon, San Jose".
    # _CAP_WORD  = un cuvant care incepe cu majuscula;
    # _CONN_WORD = particule de legatura (de, del, la, y...);
    # _CITY_SPEC = numele unui oras de 1-3 cuvinte ("Santa Ana", "Montes de Oca").
    _CAP_WORD = r"[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]+"
    _CONN_WORD = r"(?:de|del|la|las|los|y|al|a|el)"
    _CITY_SPEC = rf"{_CAP_WORD}(?:\s+(?:{_CONN_WORD}\s+)?{_CAP_WORD}){{0,2}}"
    # Cuvant cheie + prima parte, virgula, a doua parte, optional virgula + oras
    cr_addr_re = re.compile(
        rf"\b(?:Avenida|Av\.?|Calle|Paseo|Diagonal|Transversal|Boulevard|Blvd\.?|"
        rf"Barrio|Urbanizaci[oó]n|Urb\.?|Residencial|Condominio|Ruta|Carretera)\s+"
        rf"[^,.;\n!?]+,\s*[^,.;\n!?]+(?:,\s*{_CITY_SPEC})?"
    )
    text = cr_addr_re.sub(lambda regex_match: _record_address(regex_match.group(0)), text)

    # 7b. Adrese recunoscute dupa context. Grupul 1 = expresia de context (ramane in text),
    # grupul 2 = adresa (se inlocuieste). Adresa se termina la punct, la sfarsit de rand sau
    # inaintea unor cuvinte care incep alta propozitie ("and keep", "please"...).
    addr_ctx_patterns = [
        re.compile(
            # "pickup at <adresa>"
            r"(\bpickup at )(.+?)(?=\s+(?:to\s+our|for\s+our|and\s+keep|on\s+file|if\s+you|please)|[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            # "send the paperwork / invoice / reply ... to <adresa>"
            r"(\b(?:send|mail|deliver|forward)\s+(?:the\s+)?(?:paperwork|written\s+reply|reply|documents|docs|invoice|receipt|confirmation|package|letter|itinerary)\s+to\s+)(.+?)(?=\s+(?:and\s+keep|on\s+file|asap)|[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            # "my (postal / home / billing ...) address is <adresa>"
            r"(\b(?:my|our)\s+(?:postal\s+|mailing\s+|home\s+|street\s+|residential\s+|billing\s+)?address\s+is\s+)(.+?)(?=\s+(?:and\s+keep)|[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            # "I live at / we live at / residing at <adresa>"
            r"(\b(?:i\s+live\s+at|we\s+live\s+at|residing\s+at)\s+)(.+?)(?=[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
    ]
    # Inlocuim doar daca textul prins chiar arata a adresa (vezi _is_plausible_address_context)
    for ctx_re in addr_ctx_patterns:
        def _ctx_addr_repl(regex_match: re.Match[str]) -> str:
            cand = regex_match.group(2).strip()
            if _is_plausible_address_context(cand):
                return regex_match.group(1) + _record_address(cand)
            return regex_match.group(0)
        text = ctx_re.sub(_ctx_addr_repl, text)

    # 7c. Bucati de adresa care apar singure, fara expresie de context
    standalone_addr_patterns = [
        # "Calle 5", "Avenida Central", "Ruta 27"
        re.compile(r"\b(?:Avenida|Av\.?|Calle|Paseo|Ruta)\s+(?:\d+[A-Za-z]?|Central)\b", flags=re.IGNORECASE),
        re.compile(
            # "200 metros al norte de la Iglesia ..." (reper dat prin distanta si directie)
            rf"\b\d+\s*(?:m|mts|metros)\s+(?:al\s+)?(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?\s+"
            rf"(?:la\s+|el\s+|los\s+|las\s+)?{_CITY_SPEC}"
        ),
        re.compile(
            # "Frente al Banco ...", "Contiguo a ...", "Costado sur de ...", "Detras del ..."
            rf"\b(?:Frente\s+(?:a|al|a\s+la)|Contiguo\s+(?:a|al|a\s+la)|"
            rf"Diagonal\s+(?:a|al|a\s+la)|Costado\s+(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?|"
            rf"Detr[aá]s\s+de(?:l|\s+la)?)\s+(?:la\s+|el\s+|los\s+|las\s+)?{_CITY_SPEC}",
            flags=re.IGNORECASE,
        ),
        re.compile(
            # Adresa in stil american: "742 Evergreen Street, Apt 3"
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\.?"
            r"(?:\s*,?\s*(?:Apt|Suite|Unit|#)\s*[A-Za-z0-9-]+)?\b"
        ),
    ]
    for addr_pat in standalone_addr_patterns:
        text = addr_pat.sub(lambda regex_match: _record_address(regex_match.group(0)), text)

    # 7d. Trecerea de variante: orice fragment distinctiv al unei adrese gasite, aparut din
    # nou in mesaj -> [ADDRESS]. Sarim peste orasele simple si peste duplicate.
    seen_frag_norms: set[str] = set()
    for frag in sorted(detected_addr_fragments, key=len, reverse=True):
        fnorm = strip_accents_lower(frag)
        if not fnorm or fnorm in seen_frag_norms or fnorm in ADDRESS_CITY_NAMES:
            continue
        seen_frag_norms.add(fnorm)
        flex = r"\s+".join(re.escape(part) for part in fnorm.split())
        pat = re.compile(rf"(?<![a-z0-9]){flex}(?![a-z0-9])")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[ADDRESS]"))

    # Etapa 8: nume de persoane -> [NAME].
    # 8a. Intai protejam numele agentiei si destinatiile turistice: le mutam in seif cu
    # textul lor original, ca regulile de nume sa nu le confunde cu persoane.
    for phrase in sorted(PROTECTED_NON_PII_PHRASES, key=len, reverse=True):
        pnorm = strip_accents_lower(phrase)
        flex = r"\s+".join(re.escape(word) for word in pnorm.split())
        pat = re.compile(rf"\b{flex}\b")
        text = sub_normalized(text, pat, lambda orig, _match: vault.stash(orig))

    # 8b. Nume gasite dupa context.
    # _NAME_WORD = un cuvant cu majuscula (poate contine accente, apostrof, cratima);
    # _NAME_SPAN = 1-4 astfel de cuvinte, cu particule permise intre ele ("de los", "del", "y").
    _NAME_WORD = r"[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ'-]+"
    _NAME_SPAN = rf"{_NAME_WORD}(?:\s+(?:(?:de(?:\s+l[ao]s)?|del|y)\s+)?{_NAME_WORD}){{0,3}}"

    # Rude si insotitori: "my wife Ana", "my son Diego"
    _RELATIVES = (
        r"son|daughter|wife|husband|partner|spouse|brother|sister|mother|father|"
        r"mom|dad|parent|child|kid|friend|companion|colleague|cousin|uncle|aunt|"
        r"nephew|niece|grandfather|grandmother|grandpa|grandma|fianc[eé]e?|guest|"
        r"traveler|traveller|passenger"
    )
    # Formule de incheiere, dupa care urmeaza semnatura pe randul urmator
    _SIGNOFFS = (
        r"Regards|Best regards|Kind regards|Warm regards|Sincerely|Yours sincerely|"
        r"Yours truly|Best|Thanks|Thank you|Many thanks|Cheers|Respectfully|"
        r"Cordially|Warmly|Saludos|Cordialmente|Atentamente|Gracias"
    )
    # Titluri de politete: "Mr. Smith", "Sra. Vargas", "Doña Rosa"
    _TITLES = r"Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Sr\.?|Sra\.?|Srta\.?|Don|Doña"

    # Inregistram numele (pentru 8d) si pastram contextul, inlocuind doar numele
    def _name_ctx_repl(prefix: str, raw_name: str) -> str:
        _add_name_candidate(raw_name, doc_full_names, doc_name_tokens)
        return prefix + vault.stash("[NAME]")

    # Tiparele de context: grupul 1 = contextul, grupul 2 = numele. "(?i:...)" face doar
    # contextul insensibil la majuscule; numele trebuie sa inceapa cu majuscula.
    name_ctx_patterns: list[re.Pattern[str]] = [
        # "my name is", "this is", "I am", "I'm" + nume
        re.compile(rf"(\b(?i:my\s+(?:full\s+)?name\s+is|name\s+is|this\s+is|i\s+am|i'm)\s+)({_NAME_SPAN})\b"),
        # "my wife / son / friend ..." + nume
        re.compile(rf"(\b(?i:my\s+(?:{_RELATIVES}))\s+)({_NAME_SPAN})\b"),
        # "traveling with", "booked under", "passenger:", "contact:" ... + nume
        re.compile(rf"(\b(?i:with\s+my\s+(?:{_RELATIVES})|accompanied\s+by|traveling\s+with|travelling\s+with|under\s+the\s+name|booked\s+(?:for|under)|passenger\s*:?|guest\s*:?|contact\s*:?|name\s*:?|signed(?:\s+by)?)\s+)({_NAME_SPAN})\b"),
        # Semnatura: "Regards," pe un rand, numele pe randul urmator
        re.compile(rf"(\b(?i:{_SIGNOFFS})[,!.]?\s*\n+\s*)({_NAME_SPAN})(?=\s*(?:[,.;!\n]|$))"),
        # Titlu + nume
        re.compile(rf"(\b(?:{_TITLES})\s+)({_NAME_SPAN})\b"),
    ]

    for npat in name_ctx_patterns:
        # Daca toate cuvintele prinse sunt cuvinte comune ("I am Very Upset"), nu e un nume
        # si lasam textul neschimbat.
        def _repl_fn(regex_match: re.Match[str]) -> str:
            cand = regex_match.group(2)
            words_norm = [strip_accents_lower(word) for word in cand.split()]
            if all(word in NON_NAME_STOPWORDS for word in words_norm):
                return regex_match.group(0)
            return _name_ctx_repl(regex_match.group(1), cand)
        text = npat.sub(_repl_fn, text)

    # 8c. Nume recunoscute din lexicoane: 2-4 cuvinte consecutive cu majuscula
    multi_cap_re = re.compile(rf"\b{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,3}}\b")

    def _lexicon_name_repl(regex_match: re.Match[str]) -> str:
        phrase = regex_match.group(0)
        words = phrase.split()
        norms = [strip_accents_lower(word) for word in words]
        # Primul cuvant e prenume cunoscut si cel putin unul din restul e nume de familie cunoscut
        if len(words) >= 2 and norms[0] in FIRST_NAMES and any(word_norm in LAST_NAMES for word_norm in norms[1:]):
            # ...si TOATE cuvintele sunt prenume sau nume de familie
            # (ca "Maria Lopez Hotel" sa nu fie luat intreg drept nume)
            if all(word_norm in FIRST_NAMES or word_norm in LAST_NAMES for word_norm in norms):
                _add_name_candidate(phrase, doc_full_names, doc_name_tokens)
                return vault.stash("[NAME]")
        return phrase

    text = multi_cap_re.sub(_lexicon_name_repl, text)

    # 8d. Orice reaparitie a numelor deja gasite, ignorand majusculele si accentele:
    # intai numele complete (cele mai lungi intai), apoi fiecare cuvant al lor singur.
    for full_norm in sorted(doc_full_names, key=len, reverse=True):
        parts = full_norm.split()
        if not parts:
            continue
        flex = r"\s+".join(re.escape(name_part) for name_part in parts)
        pat = re.compile(rf"\b{flex}\b")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[NAME]"))

    # Cuvintele de nume singure (ex. "Solano", din carlos.solano@... sau din "Carlos Solano") -> [NAME]
    for tok_norm in sorted(doc_name_tokens, key=len, reverse=True):
        if len(tok_norm) < 2 or tok_norm in NON_NAME_STOPWORDS:
            continue
        pat = re.compile(rf"\b{re.escape(tok_norm)}\b")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[NAME]"))

    # 8e. Partile locale simple de e-mail (ex. "carlos" din carlos@gmail.com), aparute
    # singure in text -> [EMAIL]
    for local_norm in sorted(email_locals_plain, key=len, reverse=True):
        if len(local_norm) < 3 or local_norm in NON_NAME_STOPWORDS:
            continue
        pat = re.compile(rf"\b{re.escape(local_norm)}\b")
        text = sub_normalized(text, pat, lambda _orig, _match: vault.stash("[EMAIL]"))

    # Etapa 9: golim seiful: referintele PV si frazele protejate revin la textul original,
    # iar marcajele placeholderelor devin [EMAIL], [NAME] etc.
    text = vault.restore(text)

    # Mai multe [NAME] consecutive (prenume si nume inlocuite separat) -> un singur [NAME]
    text = re.sub(r"\[NAME\](?:\s+\[NAME\])+", "[NAME]", text)

    # Daca nu a mai ramas nimic, intoarcem [REDACTED]
    return text if text.strip() else "[REDACTED]"


# ===== Intrare / iesire =====
# Alegem fisierul de intrare din directorul curent: test-input.json (la evaluare), altfel
# train-input.json, altfel files/train-input.json de langa script.
def resolve_io_paths() -> tuple[str, str]:
    """Alege caile fisierelor JSON de intrare si de iesire."""
    if os.path.isfile("test-input.json"):
        return "test-input.json", "test-output.json"
    if os.path.isfile("train-input.json"):
        return "train-input.json", "train-output.json"
    fallback = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "files", "train-input.json"
    )
    if os.path.isfile(fallback):
        return fallback, "train-output.json"
    sys.exit(f"No test-input.json or train-input.json found in {os.getcwd()}")


# Citeste documentele, anonimizeaza fiecare text si scrie rezultatul JSON.
def main() -> None:
    in_path, out_path = resolve_io_paths()
    with open(in_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Pentru fiecare document: id -> textul anonimizat
    documents = data.get("documents", [])
    redacted_documents: dict[str, str] = {}
    for doc in documents:
        doc_id = doc["id"]
        raw_text = doc.get("text", "")
        redacted_documents[doc_id] = redact_document(raw_text)

    # Formatul cerut: {"redacted_documents": {id: text}}; ensure_ascii=False pastreaza accentele
    result = {"redacted_documents": redacted_documents}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # Mesajul de stare merge pe stderr, ca sa nu se amestece cu rezultatul
    print(f"Processed {len(redacted_documents)} documents: {in_path} -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
