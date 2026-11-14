#!/usr/bin/env python3
"""Pura Vida, No PII (Costa Rica) — PII anonymization pipeline.

Reads test-input.json (or train-input.json / files/train-input.json fallback)
from the current working directory and writes test-output.json (or train-output.json)
in the current working directory with all PII redacted while preserving PV-#####
booking references and non-PII text needed for downstream classification.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Callable

# Private Use Area sentinel markers so protected spans / placeholders cannot
# collide with normal text or trigger subsequent PII patterns.
_SENTINEL_OPEN = "\uE000"
_SENTINEL_CLOSE = "\uE001"
_DIGIT_TO_PUA = str.maketrans("0123456789", "\uE010\uE011\uE012\uE013\uE014\uE015\uE016\uE017\uE018\uE019")


def _encode_index(idx: int) -> str:
    return str(idx).translate(_DIGIT_TO_PUA)


class Vault:
    """Stores protected non-PII spans and emitted placeholders behind PUA sentinels."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def stash(self, value: str) -> str:
        idx = len(self._items)
        self._items.append(value)
        return f"{_SENTINEL_OPEN}{_encode_index(idx)}{_SENTINEL_CLOSE}"

    def restore(self, text: str) -> str:
        pua_to_digit = {chr(0xE010 + d): str(d) for d in range(10)}
        pattern = re.compile(
            re.escape(_SENTINEL_OPEN)
            + r"([\uE010-\uE019]+)"
            + re.escape(_SENTINEL_CLOSE)
        )

        def _repl(match: re.Match[str]) -> str:
            digits = "".join(pua_to_digit[ch] for ch in match.group(1))
            return self._items[int(digits)]

        # Run twice in case a stashed value itself contained a sentinel.
        out = pattern.sub(_repl, text)
        out = pattern.sub(_repl, out)
        return out


# ---------------------------------------------------------------------------
# Normalization helpers (case + accent insensitive with exact index alignment)
# ---------------------------------------------------------------------------

def strip_accents_lower(s: str) -> str:
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").lower()


def build_normalized_index(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return (accent_stripped_lowercase_str, [(orig_start, orig_end), ...])."""
    norm_chars: list[str] = []
    spans: list[tuple[int, int]] = []
    for idx, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if cat == "Mn":
            if spans:
                s0, _ = spans[-1]
                spans[-1] = (s0, idx + 1)
            continue
        decomp = unicodedata.normalize("NFD", ch)
        base = [c.lower() for c in decomp if unicodedata.category(c) != "Mn"]
        if not base:
            continue
        for b in base:
            norm_chars.append(b)
            spans.append((idx, idx + 1))
    return "".join(norm_chars), spans


def replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping (start, end, replacement) edits on text."""
    if not spans:
        return text
    spans_sorted = sorted(spans, key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int, str]] = []
    cur_end = -1
    for start, end, repl in spans_sorted:
        if start < cur_end:
            continue
        merged.append((start, end, repl))
        cur_end = end
    parts: list[str] = []
    pos = 0
    for start, end, repl in merged:
        parts.append(text[pos:start])
        parts.append(repl)
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def sub_normalized(
    text: str,
    pattern: re.Pattern[str],
    repl_fn: Callable[[str, re.Match[str]], str],
) -> str:
    """Match pattern on accent-stripped lowercase text and replace in original text."""
    norm_text, spans = build_normalized_index(text)
    edits: list[tuple[int, int, str]] = []
    for m in pattern.finditer(norm_text):
        if m.start() == m.end():
            continue
        orig_start = spans[m.start()][0]
        orig_end = spans[m.end() - 1][1]
        orig_slice = text[orig_start:orig_end]
        replacement = repl_fn(orig_slice, m)
        if replacement != orig_slice:
            edits.append((orig_start, orig_end, replacement))
    return replace_spans(text, edits)


# ---------------------------------------------------------------------------
# Lexicons & Constants
# ---------------------------------------------------------------------------

MONTH_NAME_TO_NUM: dict[str, int] = {
    # English
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
    # Spanish
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

_MONTH_ALT = "|".join(sorted(MONTH_NAME_TO_NUM.keys(), key=len, reverse=True))

# Non-PII geographic / agency phrases that must not be redacted as person names
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

# Common city/province names in Costa Rica (non-distinctive city suffix of an address;
# should NOT be extracted as standalone address fragments outside the full address).
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

# Non-name words that might appear in email local parts or near capitalized contexts
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

# Curated Costa Rican / Hispanic / English first names (accent-stripped lowercase)
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

# Curated Costa Rican / Hispanic / common surnames (accent-stripped lowercase)
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


# ---------------------------------------------------------------------------
# Date validation and variant generation
# ---------------------------------------------------------------------------

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


def _dob_variants(year: int, month: int, day: int) -> list[str]:
    """Generate regex patterns (on normalized lowercase text) for a given date."""
    y = f"{year:04d}"
    m2 = f"{month:02d}"
    d2 = f"{day:02d}"
    m1 = str(month)
    d1 = str(day)
    m_alts = "|".join(re.escape(name) for name in NUM_TO_MONTHS[month])
    sep = r"[\s./-]"
    pats = [
        # ISO & slash / dash / dot
        rf"\b{y}{sep}0?{m1}{sep}0?{d1}\b",
        rf"\b0?{d1}{sep}0?{m1}{sep}{y}\b",
        rf"\b0?{m1}{sep}0?{d1}{sep}{y}\b",
        # Glued 8-digit & unpadded forms
        rf"\b{y}{m2}{d2}\b",
        rf"\b{d2}{m2}{y}\b",
        rf"\b{m2}{d2}{y}\b",
        # Month-name forms (spaces, optional commas, optional 'de')
        rf"\b(?:{m_alts})\.?\s*0?{d1}(?:st|nd|rd|th)?(?:\s*,\s*|\s+(?:de\s+)?){y}\b",
        rf"\b0?{d1}(?:st|nd|rd|th)?\s*(?:de\s+)?(?:{m_alts})\.?(?:\s*(?:de|,)\s*|\s+){y}\b",
        # Glued month-name forms (no spaces/separators)
        rf"\b(?:{m_alts})0?{d1}{y}\b",
        rf"\b0?{d1}(?:{m_alts}){y}\b",
    ]
    return pats


# ---------------------------------------------------------------------------
# Core PII redaction pipeline
# ---------------------------------------------------------------------------

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


def _extract_address_fragments(addr_text: str) -> list[str]:
    """Extract distinctive fragments (street, landmark/house) from an address."""
    frags: list[str] = []
    parts = [p.strip(" ,.;:") for p in addr_text.split(",") if p.strip(" ,.;:")]
    for idx, part in enumerate(parts):
        norm = strip_accents_lower(part)
        # Do not extract standalone trailing city names as global fragments
        # (cities like San José / Tamarindo / Manuel Antonio can appear in non-PII travel text).
        if idx == len(parts) - 1 and norm in ADDRESS_CITY_NAMES and len(parts) >= 2:
            continue
        if len(norm) >= 4:
            frags.append(part)
        # Also extract inner named landmark after directional prefixes
        m = re.search(
            r"(?:metros\s+(?:al\s+)?(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?|"
            r"frente\s+(?:a|al|a\s+la)|contiguo\s+(?:a|al|a\s+la)|"
            r"diagonal\s+(?:a|al|a\s+la)|costado\s+(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?|"
            r"detr[aá]s\s+de(?:l|\s+la)?)\s+(?:la\s+|el\s+|los\s+|las\s+)?(.+)",
            part,
            flags=re.IGNORECASE,
        )
        if m:
            inner = m.group(1).strip(" ,.;:")
            inner_norm = strip_accents_lower(inner)
            if len(inner_norm) >= 5 and inner_norm not in ADDRESS_CITY_NAMES:
                frags.append(inner)
    # Deduplicate keeping longest first
    unique: dict[str, str] = {}
    for f in frags:
        k = strip_accents_lower(f)
        if k and k not in unique:
            unique[k] = f
    return sorted(unique.values(), key=len, reverse=True)


def _add_name_candidate(
    raw_name: str,
    full_names: set[str],
    name_tokens: set[str],
) -> None:
    cleaned = re.sub(r"\s+", " ", raw_name).strip(" ,.;:!?\"'()[]{}")
    norm_full = strip_accents_lower(cleaned)
    if not norm_full or len(norm_full) < 2:
        return
    if norm_full in NON_NAME_STOPWORDS:
        return
    full_names.add(norm_full)
    for tok in re.split(r"[\s]+", norm_full):
        tok = tok.strip("'-")
        if len(tok) >= 2 and tok not in NON_NAME_STOPWORDS and tok not in ("de", "del", "la", "las", "los", "y"):
            name_tokens.add(tok)


def redact_document(text: str) -> str:
    if not text or not text.strip():
        return "[REDACTED]"

    vault = Vault()

    # 1. Protect booking references PV-#####
    pv_re = re.compile(r"\bPV-[A-Za-z0-9]{4,10}\b")
    text = pv_re.sub(lambda m: vault.stash(m.group(0)), text)

    # 2. Emails: redact full emails and collect local parts + candidate name tokens
    email_re = re.compile(
        r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![A-Za-z0-9])"
    )
    email_locals_sep: set[str] = set()
    email_locals_plain: set[str] = set()
    doc_name_tokens: set[str] = set()
    doc_full_names: set[str] = set()

    def _email_repl(m: re.Match[str]) -> str:
        local = m.group(1)
        local_norm = strip_accents_lower(local)
        if len(local_norm) >= 3 and local_norm not in NON_NAME_STOPWORDS:
            if re.search(r"[._+\-\d]", local_norm):
                email_locals_sep.add(local_norm)
            else:
                email_locals_plain.add(local_norm)
        # Extract name parts from local part
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

    # Redact standalone email local parts that contain separators or digits (e.g. carlos.solano)
    for local_norm in sorted(email_locals_sep, key=len, reverse=True):
        escaped = re.escape(local_norm)
        # Allow minor separator differences (., _, -) if local part uses separators
        flex = re.sub(r"\\([._+\-])", r"[._+-]", escaped)
        pat = re.compile(rf"(?<![a-z0-9._+-]){flex}(?![a-z0-9])")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[EMAIL]"))

    # 3. Bank Accounts (IBAN grouped and glued)
    iban_digit_cores: set[str] = set()

    def _record_iban(raw: str) -> str:
        alnum = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
        digits_only = re.sub(r"\D", "", alnum)
        if len(digits_only) >= 14:
            iban_digit_cores.add(digits_only)
            if len(digits_only) >= 18:
                # Core without 2-digit IBAN check digits
                iban_digit_cores.add(digits_only[2:])
        return vault.stash("[ACCOUNT]")

    # Costa Rica IBAN (CR + 20 digits, optional spaces/hyphens)
    cr_iban_re = re.compile(r"\b[Cc][Rr](?:[\s-]?\d){20}\b")
    text = cr_iban_re.sub(lambda m: _record_iban(m.group(0)), text)

    # General uppercase IBAN (2 letters + 2 digits + 11..30 alphanumeric chars)
    gen_iban_re = re.compile(
        r"\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]{4}){2,7}(?:[\s-]?[A-Z0-9]{1,4})?\b"
    )

    def _gen_iban_repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        compact = re.sub(r"[^A-Z0-9]", "", raw)
        if 15 <= len(compact) <= 34 and sum(ch.isdigit() for ch in compact) >= 8:
            return _record_iban(raw)
        return raw

    text = gen_iban_re.sub(_gen_iban_repl, text)

    # Redact any remaining occurrence of detected IBAN digits (with/without CR and separators)
    for core in sorted(iban_digit_cores, key=len, reverse=True):
        flex_digits = r"[\s-]*".join(core)
        pat = re.compile(rf"(?<![a-z0-9])(?:[a-z]{{2}}[\s-]*)?{flex_digits}(?!\d)")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[ACCOUNT]"))

    # 4. Dates of Birth (all formats + cross-format variants)
    detected_dobs: set[tuple[int, int, int]] = set()

    def _stash_dob(year: int, month: int, day: int) -> str:
        detected_dobs.add((year, month, day))
        return vault.stash("[DOB]")

    # 4a. Month-name formats: "March 26, 1975" / "26 March 1975" / "26 de marzo de 1975"
    dob_month_first_re = re.compile(
        rf"\b(?P<mon>{_MONTH_ALT})\.?\s+(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
        rf"(?:\s*,\s*|\s+(?:de\s+)?)(?P<year>19\d{{2}}|20[0-2]\d)\b"
    )

    def _dob_mf_repl(orig: str, m: re.Match[str]) -> str:
        y = int(m.group("year"))
        mon = MONTH_NAME_TO_NUM[m.group("mon")]
        d = int(m.group("day"))
        if _is_valid_date(y, mon, d):
            return _stash_dob(y, mon, d)
        return orig

    text = sub_normalized(text, dob_month_first_re, _dob_mf_repl)

    dob_day_first_re = re.compile(
        rf"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:de\s+)?"
        rf"(?P<mon>{_MONTH_ALT})\.?(?:\s*(?:de|,)\s*|\s+)(?P<year>19\d{{2}}|20[0-2]\d)\b"
    )

    def _dob_df_repl(orig: str, m: re.Match[str]) -> str:
        y = int(m.group("year"))
        mon = MONTH_NAME_TO_NUM[m.group("mon")]
        d = int(m.group("day"))
        if _is_valid_date(y, mon, d):
            return _stash_dob(y, mon, d)
        return orig

    text = sub_normalized(text, dob_day_first_re, _dob_df_repl)

    # 4b. ISO numeric: YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
    dob_iso_re = re.compile(
        r"\b(?P<year>19\d{2}|20[0-2]\d)([./-])(?P<mon>0?[1-9]|1[0-2])\2(?P<day>0?[1-9]|[12]\d|3[01])\b"
    )

    def _dob_iso_repl(orig: str, m: re.Match[str]) -> str:
        y = int(m.group("year"))
        mon = int(m.group("mon"))
        d = int(m.group("day"))
        if _is_valid_date(y, mon, d):
            return _stash_dob(y, mon, d)
        return orig

    text = sub_normalized(text, dob_iso_re, _dob_iso_repl)

    # 4c. Numeric DD/MM/YYYY or MM/DD/YYYY (with / or - or .)
    dob_dmy_re = re.compile(
        r"\b(?P<p1>0?[1-9]|[12]\d|3[01])([./-])(?P<p2>0?[1-9]|[12]\d|3[01])\2(?P<year>19\d{2}|20[0-2]\d)\b"
    )

    def _dob_dmy_repl(orig: str, m: re.Match[str]) -> str:
        p1 = int(m.group("p1"))
        p2 = int(m.group("p2"))
        y = int(m.group("year"))
        # Prefer DD/MM/YYYY (Costa Rican standard); fall back to MM/DD/YYYY if p2 > 12
        if p2 <= 12 and _is_valid_date(y, p2, p1):
            if p1 <= 12 and p1 != p2 and _is_valid_date(y, p1, p2):
                # Record ambiguous alternate too so cross-format variants catch both
                detected_dobs.add((y, p1, p2))
            return _stash_dob(y, p2, p1)
        if p1 <= 12 and _is_valid_date(y, p1, p2):
            return _stash_dob(y, p1, p2)
        return orig

    text = sub_normalized(text, dob_dmy_re, _dob_dmy_repl)

    # 4d. Glued 8-digit DOB explicitly in birth-date context (e.g. "date of birth is 19750326")
    dob_glued_ctx_re = re.compile(
        r"\b(?:date of birth|dob|born|nacimiento)\s*(?:is\s+|:?\s*)(?P<digits>\d{8})\b"
    )

    def _dob_glued_repl(orig: str, m: re.Match[str]) -> str:
        digs = m.group("digits")
        prefix = orig[:-len(digs)]
        # Check YYYYMMDD then DDMMYYYY then MMDDYYYY
        y1, m1, d1 = int(digs[:4]), int(digs[4:6]), int(digs[6:8])
        if _is_valid_date(y1, m1, d1):
            return prefix + _stash_dob(y1, m1, d1)
        d2, m2, y2 = int(digs[:2]), int(digs[2:4]), int(digs[4:8])
        if _is_valid_date(y2, m2, d2):
            return prefix + _stash_dob(y2, m2, d2)
        m3, d3, y3 = int(digs[:2]), int(digs[2:4]), int(digs[4:8])
        if _is_valid_date(y3, m3, d3):
            return prefix + _stash_dob(y3, m3, d3)
        return orig

    text = sub_normalized(text, dob_glued_ctx_re, _dob_glued_repl)

    # 4e. Cross-variant pass: eliminate any other representation of detected DOBs
    for y, mon, d in sorted(detected_dobs):
        for pat_str in _dob_variants(y, mon, d):
            text = sub_normalized(
                text,
                re.compile(pat_str),
                lambda _orig, _m: vault.stash("[DOB]"),
            )

    # 5. Phones (international first, then cédulas, then national 8-digit phones,
    #    then cross-variant phone numbers)
    detected_phone_cores: set[str] = set()

    def _record_phone(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 11 and digits.startswith("506"):
            detected_phone_cores.add(digits[3:])
        elif len(digits) == 8:
            detected_phone_cores.add(digits)
        elif len(digits) >= 7:
            detected_phone_cores.add(digits)
        return vault.stash("[PHONE]")

    # 5a. Costa Rica international phone (+506 / 00506 / 506)
    cr_intl_phone_re = re.compile(
        r"(?<![A-Za-z0-9])(?:\+|00)?\s*\(?\s*506\s*\)?[\s.-]*[2-8]\d{3}[\s.-]*\d{4}\b"
    )
    text = cr_intl_phone_re.sub(lambda m: _record_phone(m.group(0)), text)

    # 5b. General international phone numbers (e.g. +1 555-123-4567, +34 612 345 678)
    gen_intl_phone_re = re.compile(
        r"(?<![A-Za-z0-9])\+\d{1,3}[\s.-]*(?:\(\d{1,4}\)[\s.-]*)?\d{2,4}(?:[\s.-]*\d{2,4}){1,3}\b"
    )

    def _gen_intl_phone_repl(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 8 <= len(digits) <= 15:
            return _record_phone(m.group(0))
        return m.group(0)

    text = gen_intl_phone_re.sub(_gen_intl_phone_repl, text)

    # 6. National IDs / Cédulas (hyphenated/spaced and glued 9-digit, plus DIMEX/jurídica in ID context)
    detected_cedulas: set[str] = set()

    def _record_cedula(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        if digits:
            detected_cedulas.add(digits)
        return vault.stash("[ID]")

    # Standard Costa Rican 9-digit cédula (X-XXXX-XXXX or X XXXX XXXX)
    cedula_sep_re = re.compile(r"\b[1-9][\s-]\d{4}[\s-]\d{4}\b")
    text = cedula_sep_re.sub(lambda m: _record_cedula(m.group(0)), text)

    # Cédula jurídica (X-XXX-XXXXXX)
    cedula_jur_re = re.compile(r"\b[1-9]-\d{3}-\d{6}\b")
    text = cedula_jur_re.sub(lambda m: _record_cedula(m.group(0)), text)

    # Glued 9-digit cédula anywhere (or 10-12 digit DIMEX/ID near ID context)
    cedula_ctx_re = re.compile(
        r"\b(?:cedula|identidad|identity|dimex|id)\s*(?:number\s+is\s+|is\s+|#?\s*:?\s*)(?P<ced>\d{9,12})\b"
    )
    text = sub_normalized(
        text,
        cedula_ctx_re,
        lambda orig, m: orig[:-len(m.group("ced"))] + _record_cedula(m.group("ced")),
    )

    cedula_glued_re = re.compile(r"(?<![$₡€£\d])\b[1-9]\d{8}\b(?![\d%])")
    text = cedula_glued_re.sub(lambda m: _record_cedula(m.group(0)), text)

    # Cross-variant pass for detected cédulas (e.g. 1-2345-6789 <-> 123456789)
    for ced in sorted(detected_cedulas, key=len, reverse=True):
        if len(ced) == 9:
            p1, p2, p3 = ced[0], ced[1:5], ced[5:9]
            pat = re.compile(rf"\b{p1}[\s.-]*{p2}[\s.-]*{p3}\b")
        else:
            flex = r"[\s.-]*".join(ced)
            pat = re.compile(rf"\b{flex}\b")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[ID]"))

    # 5c. National 8-digit phones (with separator or glued)
    cr_nat_sep_phone_re = re.compile(r"(?<![$₡€£\d])\b[2-8]\d{3}[\s.-]\d{4}\b(?![\d%])")
    text = cr_nat_sep_phone_re.sub(lambda m: _record_phone(m.group(0)), text)

    us_phone_re = re.compile(r"(?<![A-Za-z0-9])\(\d{3}\)[\s.-]*\d{3}[\s.-]*\d{4}\b")
    text = us_phone_re.sub(lambda m: _record_phone(m.group(0)), text)

    cr_nat_glued_phone_re = re.compile(r"(?<![$₡€£\d])\b[2-8]\d{7}\b(?![\d%])")
    text = cr_nat_glued_phone_re.sub(lambda m: _record_phone(m.group(0)), text)

    # Cross-variant pass for detected phone numbers
    for core in sorted(detected_phone_cores, key=len, reverse=True):
        if len(core) == 8:
            p1, p2 = core[:4], core[4:]
            pat = re.compile(
                rf"(?<![a-z0-9])(?:\+?\s*0?0?\s*506[\s.-]*)?{p1}[\s.-]*{p2}(?!\d)"
            )
        else:
            flex = r"[\s.-]*".join(core)
            pat = re.compile(rf"(?<![a-z0-9])\+?\s*{flex}(?!\d)")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[PHONE]"))

    # 7. Postal Addresses + distinctive fragments
    detected_addr_fragments: list[str] = []

    def _record_address(raw: str) -> str:
        detected_addr_fragments.extend(_extract_address_fragments(raw))
        return vault.stash("[ADDRESS]")

    # 7a. Costa Rican 3-part (or 2-part) address starting with street/barrio keyword
    _CAP_WORD = r"[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]+"
    _CONN_WORD = r"(?:de|del|la|las|los|y|al|a|el)"
    _CITY_SPEC = rf"{_CAP_WORD}(?:\s+(?:{_CONN_WORD}\s+)?{_CAP_WORD}){{0,2}}"
    cr_addr_re = re.compile(
        rf"\b(?:Avenida|Av\.?|Calle|Paseo|Diagonal|Transversal|Boulevard|Blvd\.?|"
        rf"Barrio|Urbanizaci[oó]n|Urb\.?|Residencial|Condominio|Ruta|Carretera)\s+"
        rf"[^,.;\n!?]+,\s*[^,.;\n!?]+(?:,\s*{_CITY_SPEC})?"
    )
    text = cr_addr_re.sub(lambda m: _record_address(m.group(0)), text)

    # 7b. Contextual delivery / pickup / address phrases
    addr_ctx_patterns = [
        re.compile(
            r"(\bpickup at )(.+?)(?=\s+(?:to\s+our|for\s+our|and\s+keep|on\s+file|if\s+you|please)|[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(\b(?:send|mail|deliver|forward)\s+(?:the\s+)?(?:paperwork|written\s+reply|reply|documents|docs|invoice|receipt|confirmation|package|letter|itinerary)\s+to\s+)(.+?)(?=\s+(?:and\s+keep|on\s+file|asap)|[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(\b(?:my|our)\s+(?:postal\s+|mailing\s+|home\s+|street\s+|residential\s+|billing\s+)?address\s+is\s+)(.+?)(?=\s+(?:and\s+keep)|[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(\b(?:i\s+live\s+at|we\s+live\s+at|residing\s+at)\s+)(.+?)(?=[.;!\n]|$)",
            flags=re.IGNORECASE,
        ),
    ]
    for ctx_re in addr_ctx_patterns:
        def _ctx_addr_repl(m: re.Match[str]) -> str:
            cand = m.group(2).strip()
            if _is_plausible_address_context(cand):
                return m.group(1) + _record_address(cand)
            return m.group(0)
        text = ctx_re.sub(_ctx_addr_repl, text)

    # 7c. Standalone street / landmark / English street phrases
    standalone_addr_patterns = [
        re.compile(r"\b(?:Avenida|Av\.?|Calle|Paseo|Ruta)\s+(?:\d+[A-Za-z]?|Central)\b", flags=re.IGNORECASE),
        re.compile(
            rf"\b\d+\s*(?:m|mts|metros)\s+(?:al\s+)?(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?\s+"
            rf"(?:la\s+|el\s+|los\s+|las\s+)?{_CITY_SPEC}"
        ),
        re.compile(
            rf"\b(?:Frente\s+(?:a|al|a\s+la)|Contiguo\s+(?:a|al|a\s+la)|"
            rf"Diagonal\s+(?:a|al|a\s+la)|Costado\s+(?:norte|sur|este|oeste)\s+de(?:l|\s+la)?|"
            rf"Detr[aá]s\s+de(?:l|\s+la)?)\s+(?:la\s+|el\s+|los\s+|las\s+)?{_CITY_SPEC}",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\.?"
            r"(?:\s*,?\s*(?:Apt|Suite|Unit|#)\s*[A-Za-z0-9-]+)?\b"
        ),
    ]
    for addr_pat in standalone_addr_patterns:
        text = addr_pat.sub(lambda m: _record_address(m.group(0)), text)

    # 7d. Cross-variant pass: redact any repeated distinctive fragment from detected addresses
    seen_frag_norms: set[str] = set()
    for frag in sorted(detected_addr_fragments, key=len, reverse=True):
        fnorm = strip_accents_lower(frag)
        if not fnorm or fnorm in seen_frag_norms or fnorm in ADDRESS_CITY_NAMES:
            continue
        seen_frag_norms.add(fnorm)
        flex = r"\s+".join(re.escape(part) for part in fnorm.split())
        pat = re.compile(rf"(?<![a-z0-9]){flex}(?![a-z0-9])")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[ADDRESS]"))

    # 8. Person Names
    # 8a. Protect known non-PII capitalized phrases (Pura Vida Voyages, tourist destinations)
    for phrase in sorted(PROTECTED_NON_PII_PHRASES, key=len, reverse=True):
        pnorm = strip_accents_lower(phrase)
        flex = r"\s+".join(re.escape(w) for w in pnorm.split())
        pat = re.compile(rf"\b{flex}\b")
        text = sub_normalized(text, pat, lambda orig, _m: vault.stash(orig))

    # 8b. Extract & redact contextual person names
    _NAME_WORD = r"[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ'-]+"
    _NAME_SPAN = rf"{_NAME_WORD}(?:\s+(?:(?:de(?:\s+l[ao]s)?|del|y)\s+)?{_NAME_WORD}){{0,3}}"

    _RELATIVES = (
        r"son|daughter|wife|husband|partner|spouse|brother|sister|mother|father|"
        r"mom|dad|parent|child|kid|friend|companion|colleague|cousin|uncle|aunt|"
        r"nephew|niece|grandfather|grandmother|grandpa|grandma|fianc[eé]e?|guest|"
        r"traveler|traveller|passenger"
    )
    _SIGNOFFS = (
        r"Regards|Best regards|Kind regards|Warm regards|Sincerely|Yours sincerely|"
        r"Yours truly|Best|Thanks|Thank you|Many thanks|Cheers|Respectfully|"
        r"Cordially|Warmly|Saludos|Cordialmente|Atentamente|Gracias"
    )
    _TITLES = r"Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Sr\.?|Sra\.?|Srta\.?|Don|Doña"

    def _name_ctx_repl(prefix: str, raw_name: str) -> str:
        _add_name_candidate(raw_name, doc_full_names, doc_name_tokens)
        return prefix + vault.stash("[NAME]")

    name_ctx_patterns: list[re.Pattern[str]] = [
        re.compile(rf"(\b(?i:my\s+(?:full\s+)?name\s+is|name\s+is|this\s+is|i\s+am|i'm)\s+)({_NAME_SPAN})\b"),
        re.compile(rf"(\b(?i:my\s+(?:{_RELATIVES}))\s+)({_NAME_SPAN})\b"),
        re.compile(rf"(\b(?i:with\s+my\s+(?:{_RELATIVES})|accompanied\s+by|traveling\s+with|travelling\s+with|under\s+the\s+name|booked\s+(?:for|under)|passenger\s*:?|guest\s*:?|contact\s*:?|name\s*:?|signed(?:\s+by)?)\s+)({_NAME_SPAN})\b"),
        re.compile(rf"(\b(?i:{_SIGNOFFS})[,!.]?\s*\n+\s*)({_NAME_SPAN})(?=\s*(?:[,.;!\n]|$))"),
        re.compile(rf"(\b(?:{_TITLES})\s+)({_NAME_SPAN})\b"),
    ]

    for npat in name_ctx_patterns:
        def _repl_fn(m: re.Match[str]) -> str:
            cand = m.group(2)
            # Avoid matching obvious non-name sentence starters or common words
            words_norm = [strip_accents_lower(w) for w in cand.split()]
            if all(w in NON_NAME_STOPWORDS for w in words_norm):
                return m.group(0)
            return _name_ctx_repl(m.group(1), cand)
        text = npat.sub(_repl_fn, text)

    # 8c. Lexicon-based <FirstName> <LastName> (+ optional middle/second surname)
    multi_cap_re = re.compile(rf"\b{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,3}}\b")

    def _lexicon_name_repl(m: re.Match[str]) -> str:
        phrase = m.group(0)
        words = phrase.split()
        norms = [strip_accents_lower(w) for w in words]
        if len(words) >= 2 and norms[0] in FIRST_NAMES and any(n in LAST_NAMES for n in norms[1:]):
            # Ensure all words look like names (either first or last name, or not in stopwords)
            if all(n in FIRST_NAMES or n in LAST_NAMES for n in norms):
                _add_name_candidate(phrase, doc_full_names, doc_name_tokens)
                return vault.stash("[NAME]")
        return phrase

    text = multi_cap_re.sub(_lexicon_name_repl, text)

    # 8d. Pass for any remaining occurrences of full names and standalone name tokens
    #     (ignoring case and accents) anywhere in the document!
    for full_norm in sorted(doc_full_names, key=len, reverse=True):
        parts = full_norm.split()
        if not parts:
            continue
        flex = r"\s+".join(re.escape(p) for p in parts)
        pat = re.compile(rf"\b{flex}\b")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[NAME]"))

    for tok_norm in sorted(doc_name_tokens, key=len, reverse=True):
        if len(tok_norm) < 2 or tok_norm in NON_NAME_STOPWORDS:
            continue
        pat = re.compile(rf"\b{re.escape(tok_norm)}\b")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[NAME]"))

    # 8e. Pass for remaining standalone single-word email local parts (e.g. "carlos" from carlos@...)
    for local_norm in sorted(email_locals_plain, key=len, reverse=True):
        if len(local_norm) < 3 or local_norm in NON_NAME_STOPWORDS:
            continue
        pat = re.compile(rf"\b{re.escape(local_norm)}\b")
        text = sub_normalized(text, pat, lambda _orig, _m: vault.stash("[EMAIL]"))

    # 9. Restore all protected spans and placeholders
    text = vault.restore(text)

    # Collapse consecutive identical [NAME] placeholders (e.g. "[NAME] [NAME]" -> "[NAME]")
    text = re.sub(r"\[NAME\](?:\s+\[NAME\])+", "[NAME]", text)

    return text if text.strip() else "[REDACTED]"


# ---------------------------------------------------------------------------
# I/O entrypoint
# ---------------------------------------------------------------------------

def resolve_io_paths() -> tuple[str, str]:
    """Select input and output JSON paths in the current working directory."""
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


def main() -> None:
    in_path, out_path = resolve_io_paths()
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    redacted_documents: dict[str, str] = {}
    for doc in documents:
        doc_id = doc["id"]
        raw_text = doc.get("text", "")
        redacted_documents[doc_id] = redact_document(raw_text)

    result = {"redacted_documents": redacted_documents}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(redacted_documents)} documents: {in_path} -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
