# Méthodologie mathématique

## 1. Dixon-Coles (1997)

Modèle de Poisson bivarié pour les scores. Pour un match domicile `h` contre
extérieur `a` :

```
log λ_h = μ + γ + att[h] − def[a]
log λ_a = μ     + att[a] − def[h]
```

- `att` : force offensive (log-échelle), `def` : solidité défensive
  (retranchée aux buts adverses)
- `γ` : avantage du terrain estimé (≈ 0,13 à 0,28 selon les ligues)
- `τ(x, y)` : correction des scores faibles

```
τ(0,0)=1−λ_hλ_aρ   τ(0,1)=1+λ_hρ   τ(1,0)=1+λ_aρ   τ(1,1)=1−ρ
```

Estimation par maximum de vraisemblance **pondéré dans le temps** :

```
L = Σ_i w_i · [ −λ_h − λ_a + x_i log λ_h + y_i log λ_a + log τ_i ]
w_i = exp(−ξ · Δjours_i) / moyenne(w)          ξ = 0,0019  (demi-vie ≈ 1 an)
```

Pénalité L2 sur `att`/`def` (`ridge = 0,035`) : indispensable pour les promus
et les équipes à faible historique, qui seraient sinon estimées sur 3 matchs.
Contrainte d'identification `moyenne(att) = 0`. Les paramètres pour une
équipe inconnue valent 0 (= équipe moyenne de la division).

Optimisation L-BFGS-B (bornes par paramètre) : ~0,5 s par ligue.

## 2. Elo

Elo maison, mis à jour après chaque match :

```
E_h = 1 / (1 + 10^(−(R_h + 62 − R_a)/400))
ΔR  = K · m · (S − E_h),   K = 20
m   = 1 si |Δbuts| ≤ 1 ; 1,5 si 2 ; (11+|Δbuts|)/8 sinon
```

Entre deux saisons, régression vers la moyenne (les effectifs changent) :
`R ← 1500 + 0,72 (R − 1500)`. L'Elo sert de feature au GBM et de contrôle de
cohérence pour les ligues où Dixon-Coles manque de données.

## 3. Gradient boosting (LightGBM)

Deux têtes entraînées sur **36 000+ matchs** de 12 ligues (mutualisées, avec
`league_id` catégoriel) :

1. classifieur multiclasse (H/D/A) — 450 arbres, `lr = 0,028`, 24 feuilles
2. deux régresseurs Poisson (buts domicile / extérieur)

Pondération temporelle `exp(−0,0008 · jours)` (demi-vie ≈ 2,4 ans).

**57 features** : Elo (3), forme 5/10 matchs (8), buts pour/contre (8),
spécificité domicile/extérieur (4), xG (4), tirs cadrés (4), repos et
congestion (4), contexte de saison (4), ligue (1).

La matrice du GBM est reconstruite à partir des λ puis recalée sur les
probabilités 1X2 du classifieur.

## 4. Ensemble et calibration

Mélange **au niveau des matrices** (pas des probabilités 1X2) :

```
M = w · M_dixon_coles + (1 − w) · M_gbm
```

- `w` : balayage 0 → 1 (pas de 0,05), minimisation de la log loss sur la
  saison de validation hors échantillon. Valeur apprise : **w ≈ 0,30**.
- Température `T` : correction de sur/sous-confiance,
  `p_k ← p_k^(1/T)` puis renormalisation et recalage de la matrice.
  Valeur apprise : **T ≈ 0,98** (bornée à [0,85 ; 1,30] pour éviter
  l'ajustement sur du bruit).

## 5. Retrait de la marge (de-vigging)

Cotes décimales `o_i`, probabilités implicites `π_i = 1/o_i`, surrond
`S = Σπ_i > 1`. Quatre estimateurs implémentés :

| Méthode | Formule | Remarque |
|---|---|---|
| multiplicative | `p_i = π_i / S` | simple, surestime les favoris |
| additive | `p_i = π_i − (S−1)/n` | répartit la marge |
| **power** (défaut) | `Σ p_i^k = 1`, `p_i ∝ π_i^k` | meilleur compromis empirique |
| shin | modèle de Shin (1993), `z` | corrige le biais favori-longshot |

## 6. Décision et mise

```
p_fair = (1 − λ) · p_modèle + λ · p_sharp        λ = 0,70
edge   = p_fair · cote − 1
```

Le pari n'est pris que si `min_edge ≤ edge ≤ max_edge` et si
`p_fair / p_sharp ∈ [0,75 ; 1,35]`.

Kelly fractionnaire :

```
f* = (p·b − q)/b,  b = cote − 1
mise = f* · 0,22 · bankroll   (bornée à 2 % de bankroll)
```

Plafonds : 7 % d'exposition par jour, 8 paris par jour, 1 pari par match.

## 7. Backtest walk-forward

Pour chaque saison testée `S` :

1. train sur toutes les saisons `< S−1`
2. calibration (w, T) sur `S−1`
3. simulation de `S` jour par jour, dans l'ordre chronologique

Hypothèses **pessimistes** :

- cote jouée = **moyenne** du marché avant clôture, moins **1,5 %** de slippage
  (les cotes de clôture ne sont pas accessibles en pratique) ;
- référence « prix juste » = clôture Pinnacle dé-viggée ;
- pas de pari si aucune cote exploitable.

Métriques : log loss, Brier, RPS, précision, ROI, yield, drawdown max,
Sharpe par pari, facteur de profit, plus longue série perdante et **CLV**
(`cote jouée / cote de clôture − 1`).

## 8. Limites connues

1. **Pas de compositions ni de blessures** : la principale information que le
   marché possède et pas nous. C'est la raison pour laquelle la probabilité
   est ancrée sur le prix sharp.
2. **xG partiel** : les colonnes xG de football-data ne commencent qu'en
   2025/26 (~11 % de l'historique).
3. **Cotes pré-clôture** : le backtest suppose qu'on peut jouer la moyenne du
   marché ; en pratique, il faut plusieurs comptes (line shopping).
4. **Variance** : 480 paris/saison, écart-type du ROI ≈ 4,5 % → une saison
   ne prouve rien. Le CLV est un indicateur plus précoce que le ROI.
5. **Sélection de championnat et de marché** : over/under et handicaps sont
   désactivés par défaut car instables sur des données de cotes gratuites.

## 9. Références

- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market*. JRSS-C.
- Shin, H. S. (1993). *Measuring the incidence of insider trading in a market
  for state-contingent claims*. Economic Journal.
- Kelly, J. L. (1956). *A New Interpretation of Information Rate*. Bell System
  Technical Journal.
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*.
