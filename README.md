# ATHENA — moteur quantitatif de pronostics football

**ATHENA** est un moteur de pronostics sportifs complet, reconstruit de zéro :
ingestion de données, modèles statistiques, détection de valeur, gestion de
bankroll, backtest honnête, publication automatique. **100 % gratuit** :
aucune donnée payante, aucun serveur, aucune carte bancaire — seulement
GitHub Actions et des sources publiques.

> Dashboard live : **https://arthur12arthur.github.io/athena-betting-engine/**
> (publié automatiquement 3× par jour)

---

## 1. Pourquoi un nouveau système

La plupart des « pronostiqueurs » vendent des tickets sans méthode. Un vrai
système de pari doit répondre à quatre questions, dans cet ordre :

| Question | Réponse ATHENA |
|---|---|
| Quelle est la probabilité réelle de chaque issue ? | Dixon-Coles + Elo + gradient boosting, calibrés |
| Le prix proposé est-il faux ? | dé-vigging (`power`) + ancrage sur la cote sharp |
| Combien miser ? | Kelly fractionnaire plafonné (22 %, max 2 %/pari) |
| Est-ce que ça marche vraiment ? | backtest walk-forward + **CLV** (cote de clôture battue) |

Le système ne « prédit » pas les matchs à la main : il estime une
distribution, la compare au marché, et ne prend un risque que lorsque
l'espérance est positive **et** l'écart crédible.

---

## 2. Ce que le système fait

- **12 ligues** suivies (Premier League, Championship, LaLiga, Serie A,
  Bundesliga, Ligue 1, Eredivisie, Pro League, Primeira Liga, Süper Lig,
  Premiership, Super League grecque)
- **~41 000 matchs** d'historique (résultats, xG, tirs, cotes de plusieurs
  bookmakers) depuis la saison 2016/17
- **3 modèles** fusionnés au niveau des **matrices de scores** (donc marchés
  cohérents entre eux) : 1X2, double chance, over/under, BTTS, handicap
- **Value betting** : comparaison prix proposé / prix juste, garde-fous
  anti-erreur de modèle, Kelly fractionnaire
- **Marchés joués** : `1X2` et **over/under 2,5 buts** ; le **handicap
  asiatique** (lignes entières, demi et **quart** : −0,25, −0,75…) est
  implémenté de bout en bout (modèle, prix, règlement) mais **désactivé par
  défaut** — voir §3, son edge est négatif sur les données gratuites
- **Règlement exact** de tous les marchés, demi-gains et remboursements
  inclus (handicap quart, push sur ligne entière)
- **Bankroll papier** persistante, règlement automatique des paris
- **Historique vérifiable** : chaque pronostic est archivé puis confronté au
  score réel (taux de réussite, log loss, Brier) — pas seulement les paris
- **Rapport Telegram automatique** à chaque exécution (bankroll, résultats
  réglés avec P&L, value bets du jour)
- **Résumé de run lisible** : le P&L et les résultats s'affichent directement
  dans l'onglet *Summary* de chaque exécution GitHub Actions
- **Dashboard statique** régénéré à chaque exécution, avec sections
  « Derniers résultats » et « Historique des pronostics »
- **Zéro coût** : données et cotes de clôture gratuites, calcul sur les
  runners GitHub publics

---

## 3. Résultats mesurés (pas des promesses)

Backtest **walk-forward** sur 3 saisons complètes (2024/25, 2025/26, 2026/27
en cours) — entraînement strictement antérieur à chaque saison testée,
validation hors échantillon, cotes moyennes du marché dégradées de 1,5 % de
slippage :

| Métrique | Valeur | Lecture |
|---|---|---|
| Matchs évalués | 9 029 | toutes ligues confondues |
| Log loss (1X2) | **0,992** | marché ≈ 0,971 → le modèle est proche, pas supérieur |
| Brier | 0,592 | échelle 0–2, plus bas = mieux |
| RPS | 0,201 | adapté aux issues ordonnées |
| Précision 1X2 | **51,0 %** | au-dessus du hasard (33 %) |
| Paris réglés | 1 446 | ~480 par saison |
| ROI | **−0,7 %** | ≈ point mort, honnête |
| **CLV moyen** | **+6,1 %** | on bat la cote de clôture : signal réel |
| Drawdown max | 23,6 % | la variance est réelle, le Kelly fractionnaire la contient |

Par marché, sur les deux dernières saisons complètes (mêmes règles de mise,
ancrage marché 0,70) :

| Marché | Paris | ROI | CLV | Décision |
|---|---|---|---|---|
| 1X2 | 685 | **+2,0 %** | +7,0 % | actif |
| Over/Under 2,5 | 281 | **+12,4 %** | +10,1 % | actif |
| Handicap asiatique | 246 | **−8,7 %** | −0,9 % | implémenté, **désactivé** |

Le handicap asiatique est le marché où le prix est le plus efficient : le
modèle n'y trouve aucun edge (CLV négatif, résultat stable sur toutes les
tranches testées — lignes pleines, quarts, domicile/extérieur, tranches de
cotes). Il reste dans le code, activable en une ligne
(`betting.markets: ["1X2","OU25","AH"]`), mais parier dessus serait de la
croyance, pas de la mesure.

**À lire sans se raconter d'histoires.** Le marché des paris est très
efficace : un modèle statistique alimenté par des données publiques arrive
*au niveau* du marché, pas au-dessus. Le ROI du backtest est donc proche de
zéro. Ce qui distingue un système utile d'une illusion, c'est le **CLV
positif** et la **calibration** — deux choses qu'ATHENA mesure et publie.
La rentabilité durable vient ensuite du *line shopping* (comparer plusieurs
bookmakers) et de la discipline de mise, pas d'une boule de cristal.

---

## 4. Architecture

```
athena-repo/
├── athena/
│   ├── config.py              # configuration typée + surcharges d'env
│   ├── http.py                # client HTTP (retries, cache disque)
│   ├── state.py               # bankroll + ledger (JSON atomique)
│   ├── live.py                # pipeline quotidien de bout en bout
│   ├── cli.py                 # interface en ligne de commande
│   ├── sources/               # connecteurs de données
│   │   ├── football_data_csv.py   # résultats + cotes de clôture (gratuit)
│   │   ├── espn.py                # calendriers / scores (gratuit, sans clé)
│   │   ├── pinnacle.py            # cotes sharp (API guest, sans clé)
│   │   ├── oddsapi.py             # multi-books (clé gratuite optionnelle)
│   │   └── gemini.py              # texte contextuel (clé gratuite optionnelle)
│   ├── etl/
│   │   ├── names.py           # résolution des noms d'équipes inter-sources
│   │   └── features.py        # features sans fuite du futur (Elo, forme, xG)
│   ├── models/
│   │   ├── markets.py         # matrices de scores -> probabilités de marchés
│   │   ├── dixon_coles.py     # Poisson bivarié, décroissance temporelle
│   │   ├── gbm.py             # LightGBM (1X2 + buts attendus)
│   │   └── ensemble.py        # mélange + calibration (poids, température)
│   ├── betting/
│   │   ├── devig.py           # retrait de marge (power/shin/mult./add.)
│   │   ├── value.py           # edge, cote juste, blending marché
│   │   ├── markets_offers.py  # offres réelles + référence sharp
│   │   └── staking.py         # Kelly fractionnaire + garde-fous
│   ├── backtest/
│   │   ├── engine.py          # walk-forward, simulation des paris
│   │   └── metrics.py         # logloss, Brier, RPS, ROI, drawdown, CLV
│   └── report/
│       ├── dashboard.py       # génère site/index.html
│       └── template.html      # design (CSS inline, SVG, zéro dépendance)
├── config/config.yaml         # toute la stratégie est ici
├── data/
│   ├── aliases.yaml           # noms d'équipes inter-sources (versionné)
│   └── state/ledger.json      # bankroll + paris (versionné)
├── site/                      # dashboard publié (GitHub Pages)
├── tests/                     # 55 tests (dont non-fuite du futur)
└── .github/workflows/         # CI, quotidien, backtest hebdo, Pages
```

Le détail des choix techniques est dans [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
et la mathématique dans [`docs/METHODOLOGIE.md`](docs/METHODOLOGIE.md).

---

## 5. Démarrage rapide

```bash
git clone https://github.com/arthur12arthur/athena-betting-engine.git
cd athena-betting-engine
pip install -e ".[dev]"

python -m athena.cli doctor          # vérifie les 5 sources
python -m athena.cli ingest          # télécharge l'historique
python -m athena.cli backtest --seasons 2024,2025,2026 --render
python -m athena.cli daily           # pipeline du jour + dashboard
open site/index.html
```

### Clés optionnelles (toutes gratuites)

| Secret GitHub | Rôle | Sans la clé |
|---|---|---|
| `ODDS_API_KEY` | cotes multi-bookmakers ([the-odds-api.com](https://the-odds-api.com), palier gratuit) | le système fonctionne avec Pinnacle seul, edges plafonnés, mises divisées par 2 |
| `GEMINI_API_KEY` | commentaire de match ([AI Studio](https://aistudio.google.com/apikey), palier gratuit) | aucun commentaire, prédictions inchangées |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | rapport quotidien sur Telegram | pas d'envoi, tout le reste fonctionne |
| `ATHENA_SITE_URL` (variable) | lien du dashboard dans le message | lien omis |

### Brancher Telegram en 4 minutes (gratuit)

1. Dans Telegram, écrire à **@BotFather** → `/newbot` → choisir un nom →
   BotFather renvoie un **token** (`123456:ABC-DEF…`).
2. Envoyer n'importe quel message à votre nouveau bot (ou l'ajouter à un
   groupe, puis `/start`).
3. Récupérer l'identifiant du destinataire :
   `curl "https://api.telegram.org/bot<TOKEN>/getUpdates"` → champ
   `message.chat.id`.
4. Dans GitHub : `Settings → Secrets and variables → Actions` →
   ajouter `TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID`
   (et la variable `ATHENA_SITE_URL` = l'URL GitHub Pages).

Test immédiat, sans attendre le cron :

```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python -m athena.cli telegram
```

Le message contient : bankroll et P&L, paris réglés du jour avec score et
P&L, value bets à venir, pronostics les plus tranchés, taux de réussite de
l'historique et lien du dashboard. Il est envoyé en HTML, découpé si > 3 900
caractères, et **n'est jamais bloquant** : une panne Telegram n'interrompt
pas le pipeline.

Ajout : `Settings → Secrets and variables → Actions → New repository secret`.
Aucune clé n'est jamais écrite dans le code ni dans un fichier du dépôt.

---

## 6. Automatisation (GitHub Actions, 100 % gratuit sur dépôt public)

| Workflow | Déclencheur | Effet |
|---|---|---|
| `daily.yml` | cron `05:20 / 11:20 / 17:20 UTC` | ingère les résultats, règle les paris, prédit les matchs à venir, publie et **déploie** `site/` |
| `backtest.yml` | lundi 03:00 UTC | réentraîne et rejoue 3 saisons, met à jour les métriques |
| `ci.yml` | à chaque push/PR | ruff + 84 tests + diagnostic réseau |
| `pages.yml` | manuel | redéploie le dashboard (secours) |

Chaque exécution commite ses résultats (`site/`, `data/state/`) : l'historique
du dépôt **est** l'historique du système. Aucune base de données à héberger.

---

## 7. Commandes

```bash
athena doctor          # diagnostic des sources
athena ingest          # historique + couverture des données
athena backtest        # walk-forward (options --seasons, --render)
athena daily           # pipeline complet (--no-llm, --no-publish, --no-notify)
athena publish         # régénère uniquement le dashboard
athena ratings         # classements internes par ligue
athena history         # historique des pronostics + paris réglés (--last N)
athena telegram        # envoie le rapport Telegram courant (test)
athena audit-aliases   # noms d'équipes non résolus entre sources
make test              # tests
```

Où voir les résultats :

| Support | Contenu |
|---|---|
| onglet **Summary** d'un run Actions | tableau des paris réglés (score, cote, mise, **P&L**, statut), value bets du jour, historique |
| **Telegram** | même informations, poussées automatiquement 3×/jour |
| `site/index.html` | sections « Derniers résultats », « Historique des pronostics », cotes justes over/under et handicap par match |
| `data/state/ledger.json` | source de vérité des paris (statuts `won`/`lost`/`push`/`half_won`/`half_lost`) |
| `data/state/predictions_history.json` | archive de tous les pronostics, vérifiés dès que le match est joué |

---

## 8. Ce que le système refuse de faire

- **Parier sans référence sharp** : sans prix juste crédible, il n'y a pas
  d'edge mesurable, seulement une opinion.
- **Suivre un edge absurde** : un désaccord de +25 % avec le marché est
  traité comme une erreur de modèle (plafond d'edge + garde-fou de divergence).
- **Cacher ses performances** : le backtest, les pertes, le drawdown et le
  CLV sont publiés tels quels, à chaque exécution.
- **Utiliser un LLM pour décider** : Gemini rédige du contexte, il ne produit
  jamais de probabilité.

---

## 9. Pistes d'amélioration (déjà cadrées dans le code)

1. **Compositions et blessures** (source gratuite à trouver) : le principal
   gisement d'alpha restant.
2. **xG complet** : la couverture xG de football-data ne démarre qu'en
   2025/26 ; élargir la source améliorerait les marchés de buts.
3. **Marchés de niche** (over/under, handicaps) : instables sur données
   gratuites, désactivés par défaut — à réactiver avec des cotes multi-books.
4. **Optimisation du portefeuille** : corrélation entre paris d'une même
   journée (exposition simultanée), Kelly multivarié.

---

## 10. Avertissement

Projet **éducatif et de recherche**. Il fonctionne en *paper trading* :
aucun argent réel, aucun compte de bookmaker. Aucune méthode ne garantit un
gain ; le backtest est une reconstitution historique, pas une promesse.
Les paris sportifs comportent des risques de perte, d'endettement et
d'addiction. **18+**. Si le jeu devient un problème : joueurs-info-service.fr
(France) ou l'équivalent local.

Licence MIT — voir [`LICENSE`](LICENSE).
