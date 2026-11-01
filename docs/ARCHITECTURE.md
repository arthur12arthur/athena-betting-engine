# Architecture technique

## 1. Flux de données

```
football-data.co.uk ──┐
ESPN (calendrier)   ──┼──> ingest ──> features (Elo/forme/xG) ──> modèles
Pinnacle (sharp)    ──┤                                             │
The Odds API (opt.) ──┘                                             v
                                                        matrice de scores
                                                                │
                                              ┌─────────────────┼─────────────────┐
                                              v                 v                 v
                                           marchés 1X2      O/U, BTTS, DC     cotes justes
                                              │
                                       edge vs prix sharp
                                              │
                                    Kelly fractionnaire + plafonds
                                              │
                                    ledger JSON + dashboard HTML
```

## 2. Décisions structurantes

### 2.1 Une matrice de scores comme source unique de vérité

Tout part de `P(i, j)` : probabilité que le match finisse `i–j`. Les marchés
en sont dérivés (`ScoreMatrix.market`). Avantage : quand on mélange deux
modèles ou qu'on calibre, **tous** les marchés restent cohérents (pas de
« 55 % domicile mais 48 % +0,5 but »).

`reconcile_to` permet de forcer les trois blocs 1X2 à des valeurs cibles
sans détruire la structure conditionnelle (répartition des scores dans
chaque issue).

### 2.2 Pas de fuite du futur, par construction

`etl/features.py` fait **un seul passage chronologique** : les features d'un
match sont calculées avant de mettre à jour l'état (Elo, forme, repos). Le
même code sert à l'entraînement, au backtest et à la production. Les tests
`tests/test_features_no_leak.py` vérifient qu'un résultat modifié ne change
aucune feature du match concerné.

Le backtest utilise :
- train : saisons `< S-1`
- validation (poids d'ensemble + température) : saison `S-1`, **exclue** du train
- test : saison `S`

### 2.3 Le prix est l'étalon, pas le modèle

Le modèle vaut `logloss ≈ 0,992` contre `0,971` pour le marché de clôture :
il est bon, pas omniscient. La probabilité utilisée pour décider est donc
**ancrée à 70 % sur le prix sharp** :

```
p_fair = 0.30 · p_modèle + 0.70 · p_sharp_deviggé
```

Conséquence assumée : on ne parie que là où le modèle *confirme* une
anomalie de prix. C'est ce qui rend le CLV positif.

### 2.4 Garde-fous plutôt que confiance

| Garde-fou | Valeur | Raison |
|---|---|---|
| `max_edge` | 10 % | un edge énorme = erreur de modèle ou cote erronée |
| `max_divergence` | p_modèle / p_sharp ∈ [0,75 ; 1,35] | idem, en relatif |
| `min_odds / max_odds` | 1,45 – 8,0 | pas de longshots illiquides |
| `min_prob / max_prob` | 6 % – 80 % | évite les paris extrêmes |
| `max_stake_pct` | 2 % | survie à une série de pertes |
| `max_daily_exposure_pct` | 7 % | pas de corrélation cachée sur une journée |
| `max_bets_per_day` | 8 | qualité > quantité |
| `single_source.*` | edge ≥ 5 %, mise ×0,5 | une seule source de cotes = edge moins fiable |

## 3. Modules

| Module | Rôle | Points d'extension |
|---|---|---|
| `sources/*` | connecteurs, chacun avec cache disque et repli | ajouter une source = implémenter `Fixture` + un parseur |
| `etl/names.py` | résolution inter-sources (normalisation, alias, fuzzy) | `data/aliases.yaml` |
| `etl/features.py` | features sans fuite | `FEATURES` + `_team_features` |
| `models/markets.py` | matrice -> marchés | ajouter un marché = une branche dans `market()` |
| `models/dixon_coles.py` | Poisson bivarié pondéré | `xi`, `ridge`, `train_window_days` |
| `models/gbm.py` | LightGBM 1X2 + Poisson buts | `config.model.gbm` |
| `models/ensemble.py` | mélange + calibration | `optimize_blend`, `optimize_temperature` |
| `betting/offers.py` | offres de marché unifiées (historique + live) | `offers_from_history`, `offers_from_live` |
| `betting/settlement.py` | règlement exact de tous les marchés (demi-gains, push) | `settle_pnl` |
| `betting/*` | dé-vig, edge, Kelly | `devig.METHODS`, `staking.plan_stake` |
| `history.py` | archive des pronostics + vérification au score réel | `PredictionArchive` |
| `sources/telegram.py` | rapport Telegram (HTML, non bloquant) | `daily_message`, `from_config` |
| `backtest/*` | walk-forward + métriques | `simulate()` pour tester une stratégie |
| `live.py` | orchestration quotidienne | `run_daily()` |
| `report/*` | dashboard HTML autonome | `template.html` |

## 4. Persistance

Pas de base de données à héberger :

- `data/state/ledger.json` — bankroll, paris, courbe de capital (écriture
  atomique : fichier temporaire puis `os.replace`). Idempotent : un pari déjà
  présent n'est jamais dupliqué (identifiant = hash de match + marché + sélection).
- `data/state/predictions_history.json` — archive des pronostics, complétée
  a posteriori avec le score réel (réussite, log loss, Brier). Séparée du
  ledger : elle mesure le **modèle**, pas la chance aux paris.
- `site/data/*.json` — publication (prédictions, ledger, historique,
  backtest, ratings, `run_summary.md`).
- `site/index.html` — rendu serveur complet : aucun `fetch`, aucune
  dépendance externe, fonctionne hors-ligne et sur GitHub Pages.

### 3.1 Un seul règlement pour deux mondes

`betting/settlement.py::settle_pnl(market, selection, line, stake, price, hg, ag)`
renvoie `(pnl, statut)` et gère 1X2, DC, BTTS, over/under et handicap, y
compris les lignes **quart** (le pari est coupé en deux moitiés). Le backtest
et le pipeline quotidien appellent la même fonction : aucun risque qu'un
`half_won` du live soit compté comme une perte dans les statistiques.

De la même manière, `betting/offers.py` produit un objet `Offer` unique
(`price_bet`, `price_sharp`, `p_sharp`, `line`) que le backtest remplisse
depuis football-data et le live depuis Pinnacle + multi-books : la stratégie
de sélection est écrite une seule fois (`select_bets` / `simulate`).

### 3.2 Rapport de run et notifications

`live.run_daily()` produit `site/data/run_summary.md`, affiché dans l'onglet
*Summary* de GitHub Actions (étape `$GITHUB_STEP_SUMMARY`) : résultats réglés
avec score, cote, mise et **P&L**, value bets du jour, statistiques
d'historique. Le même contenu est poussé sur Telegram
(`sources/telegram.py`) si `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` existent.
Toute erreur d'envoi est avalée et journalisée : la publication ne dépend
jamais d'un service tiers.

## 5. Coût d'exécution

| Poste | Coût |
|---|---|
| Github Actions (dépôt public) | 0 € — minutes illimitées |
| GitHub Pages | 0 € |
| football-data.co.uk / ESPN / Pinnacle guest | 0 €, sans clé |
| The Odds API / Gemini (optionnels) | 0 € au palier gratuit |
| Serveur, base de données | aucun |

Durée typique d'un run quotidien : **45 s** (entraînement complet inclus).
Backtest 3 saisons : **~2 min**.

## 6. Ajouter une ligue

1. `config/config.yaml` → bloc `leagues` (code football-data, slug ESPN,
   id Pinnacle).
2. `python -m athena.cli audit-aliases` → ajouter les noms manquants dans
   `data/aliases.yaml`.
3. `python -m athena.cli ingest` puis `daily` : la ligue est intégrée aux
   modèles (Dixon-Coles par ligue, GBM mutualisé avec `league_id`).
