# Lexicon scris de noi (cunoștințe de domeniu, EN + FR) pentru a eticheta propozițiile după hazard.
# Propozițiile fără niciun cuvânt de hazard (urgență, decese, pagube, drumuri închise...) sunt „generice":
# în corpus ele poartă corelații false cu eticheta (diferite în EN și FR), deci le eliminăm.
import re
LEX = {
 "flood": r"flood|inond|crue|ice jams?|embâcle|dikes?|digues?|overflow|débord(?!é)|water levels?|niveau de l'eau|\beaux\b|spring runoff|fonte des neiges|snowmelt|rivers? (?:rose|crested)|high water|sandbag|sacs de sable|submerg|under water|sous l'eau",
 "storm": r"tornad|hurrican|ouragan|hail|grêl|thunderstorm|orage|windstorm|tempête de vent|gusts?|rafales?|tropical|storm surge|onde de tempête|marée de tempête|downburst|rafale descendante|funnel|entonnoir|cyclon|squall|vitesse du vent|wind speed|\bwinds? (?:of|gusting)|vents? (?:de|violents)|uprooted|déracin|roofs?|toits?",
 "winter": r"snow|neige|blizzard|ice storm|tempête de verglas|verglas|freezing rain|pluie verglaçante|\bcold\b|\bfroid|poudrerie|frostbite|engelure|wind ?chill|refroidissement éolien|hivern|winter|\bice\b|glace|sub-zero|-\s?\d+ ?c\b|whiteout",
 "wildfire": r"wildfire|forest fire|feux? de forêt|incendies? de forêt|bush ?fire|brush fire|grass fire|feux? de brousse|hectares|forêt|timber|\bbois\b|smoke|fumée|water bombers?|bombardiers? d'eau|fire season|saison des feux",
 "drought": r"drought|sécheresse|heat ?wave|canicule|vague de chaleur|chaleur|\bheat\b|crops?|récoltes?|cultures|livestock|bétail|dust|poussière|wells? ran dry|puits|topsoil|terre arable|sol est emporté|sol s'envole|topsoil|soil blew|soil was blown|grain|wheat|blé|yields?|rendement|dry|sèche|\bsec\b|farm|ferme|ranch|rationn",
 "ground": r"landslides?|glissements? de terrain|avalanch|snow avalanche|earthquake|séisme|tremblement de terre|tsunami|rock ?slides?|éboulement|mud ?slides?|coulée|sinkhole|affaissement|magnitude|richter|slope|pente|hillside|flanc|cracks?|fissures?|seismic|sismique|debris flow|aftershocks?|répliques|bougeait|ground had been moving|shock was felt",
 "industrial": r"explosion|explos|factory|usine|warehouse|entrepôt|spill|déversement|chemical|chimique|\bmine\b|miners?|mineurs?|pipeline|oléoduc|gazoduc|collapse|effondr|refinery|raffinerie|gas leak|fuite de gaz|toxic|toxique|plant\b|blaze|brasier|incendie (?:se déclare|détruit|ravage)|fire (?:broke|destroyed|started|gutted)|hotel|hôtel|apartment|appartements|nursing home|wiring|câblage|power failure|panne|blackout|dam (?:failed|burst)|rupture|tank|réservoir|munitions",
 "transport": r"train|derail|déraill|collision|collid|percut|crash|écras|plane|avion|aircraft|airliner|appareil|ferry|traversier|vessel|navire|\bship|bateau|capsiz|chavir|\bbus\b|autobus|pile-?up|carambolage|sank|sink|founder|coul[eé]|sombre|cargo|freighter|passengers?|passagers?|locomotive|railway|ferroviaire|level crossing|passage à niveau|pilot|pilote|runway|piste|crew|équipage|excessive speed|vitesse excessive|human error|erreur humaine|aground|échou|truck|camion|vehicles?|véhicules?",
 "epidemic": r"outbreak|épidémi|influenza|grippe|flu\b|cholera|choléra|typhoid|typhoïde|smallpox|variole|polio|measles|rougeole|diphther|diphtérie|tubercul|scarlet|scarlatine|cases|\bcas\b|infect|vaccin|disease|maladie|sars|sras|quarantin|pandemi|pandémi|virus|bacteri|bactéri|contamin|e\. coli|hospitals? (?:were|set)|hôpitaux d'urgence|illness|symptom|iron lungs?|poumons? d'acier",
}
CLASSES = ["flood", "storm", "winter", "wildfire", "drought", "ground", "industrial", "transport", "epidemic"]
RX = {c: re.compile(LEX[c], re.IGNORECASE) for c in CLASSES}

def votes(s):
    return [len(RX[c].findall(s)) for c in CLASSES]

def sentences(t):
    return [s.strip() for s in re.split(r'(?<=[.;])\s+(?=[A-ZÀ-Ý0-9])', t) if s.strip()]

# expresii de context care nu înseamnă hazardul cuvântului (evacuare „cu avionul" etc.)
CONTEXT = re.compile(r"par avion|airlifted|flown (?:in|out|south)|envoyés par avion|évacués par avion|by air|voie aérienne", re.I)

def label_sentence(s):
    """Eticheta lexicală a unei propoziții sau None dacă e generică / ambiguă."""
    s = CONTEXT.sub(" ", s)
    v = votes(s)
    if re.search(r"avalanch", s, re.I):      # avalanșa (chiar „de zăpadă") ține de hazardul ground
        v[CLASSES.index("ground")] += 5
    m = max(v)
    if m == 0 or v.count(m) > 1:
        return None
    return CLASSES[v.index(m)]
