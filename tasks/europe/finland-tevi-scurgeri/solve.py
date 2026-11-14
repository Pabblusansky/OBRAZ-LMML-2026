# Finlanda – curățăm jurnalul de rute și reconstruim graful neorientat al orașelor.
# Utilizare: python3 -I solve.py files/routes.csv
import csv, sys, unicodedata, collections

def fold(s):
    """Normalizează un nume de oraș: fără spații, fără punct final,
    cu litere mici și fără diacritice (ä -> a, å -> a, ö -> o)."""
    s = s.strip().rstrip('.').strip().casefold()
    # NFKD desparte litera de semnul diacritic; păstrăm doar literele de bază
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


# Finlanda este bilingvă: unele birouri au scris numele suedez al orașului.
# Cheie = numele suedez normalizat, valoare = numele finlandez normalizat.
SV = {'helsingfors':'helsinki','abo':'turku','villmanstrand':'lappeenranta','inga':'inkoo',
 'bjorneborg':'pori','hango':'hanko','hogfors':'karkkila','idensalmi':'iisalmi','ikalis':'ikaalinen',
 'ilmola':'ilmajoki','jakobstad':'pietarsaari','kimitoon':'kemionsaari','kristinestad':'kristiinankaupunki',
 'kumo':'kokemaki','kyrkslatt':'kirkkonummi','malax':'maalahti','korsholm':'mustasaari','nadendal':'naantali',
 'narpes':'narpio','reso':'raisio','sjundea':'siuntio','vichtis':'vihti','virmo':'mynamaki','alavo':'alavus',
 'borga':'porvoo','brahestad':'raahe','esbo':'espoo','fredrikshamn':'hamina','grankulla':'kauniainen',
 'hyvinge':'hyvinkaa','tavastehus':'hameenlinna','kajana':'kajaani','karleby':'kokkola','kasko':'kaskinen',
 'kervo':'kerava','keuru':'keuruu','kronoby':'kruunupyy','lappo':'lapua','lahtis':'lahti','larsmo':'luoto',
 'lojo':'lohja','lovisa':'loviisa','pargas':'parainen','raseborg':'raasepori','raumo':'rauma',
 's:t karins':'kaarina','s:t michel':'mikkeli','sibbo':'sipoo','tammerfors':'tampere','tornea':'tornio',
 'tusby':'tuusula','uleaborg':'oulu','ulvsby':'ulvila','nykarleby':'uusikaarlepyy','nystad':'uusikaupunki',
 'vasa':'vaasa','vanda':'vantaa','vora':'voyri','lundo':'lieto','traskanda':'jarvenpaa','birkala':'pirkkala'}


def town(s):
    """Cheia canonică a orașului: întâi normalizăm textul, apoi traducem din suedeză."""
    f = fold(s)
    return SV.get(f, f)


# Calea către CSV: argumentul din linia de comandă sau, implicit, files/routes.csv lângă script
import os
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', 'routes.csv')
rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))

# Graf simplu, neorientat: fiecare muchie este o pereche NEORDONATĂ (frozenset),
# deci A-B și B-A sunt aceeași muchie, iar mulțimea elimină automat duplicatele.
# Tipul rutei (ferry/train/bus) nu contează: o legătură se numără o singură dată.
edges = set()
loops = 0
for r in rows:
    a, b = town(r['from_town']), town(r['to_town'])
    if a == b:              # bucla (orașul legat de el însuși) nu este o legătură reală
        loops += 1
        continue
    edges.add(frozenset((a, b)))

# Gradul fiecărui nod = numărul de legături distincte care ating orașul
deg = collections.Counter(t for e in edges for t in e)
print('orase', len(deg), 'legaturi', len(edges), 'bucle', loops)   # așteptat: 116 / 448

# Clasament descrescător după grad; flagul = primele litere ale primelor 7 orașe
top = deg.most_common(12)
print(top)
print('FLAG: SIGMOID_' + ''.join(t[0] for t, _ in top[:7]).upper())
