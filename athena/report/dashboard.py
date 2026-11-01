"""Génère le dashboard statique (site/index.html).

Contraintes de conception :
  - **zéro dépendance** : un seul fichier HTML, CSS inline, SVG inline,
    aucune requête réseau -> fonctionne sur GitHub Pages, en local, et
    même dans un aperçu hors-ligne.
  - lisible sur mobile et desktop.
  - toutes les données affichées viennent des JSON produits par le pipeline.
"""
from __future__ import annotations

import html
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..logging_conf import get_logger

log = get_logger("report.dashboard")

LEAGUE_LABEL = {
    "E0": "Premier League", "E1": "Championship", "SP1": "LaLiga", "I1": "Serie A",
    "D1": "Bundesliga", "F1": "Ligue 1", "N1": "Eredivisie", "B1": "Pro League",
    "P1": "Primeira Liga", "T1": "Super Lig", "SC0": "Premiership", "G1": "Super League",
}
SEL_FR = {"home": "1", "draw": "X", "away": "2"}


# ---------------------------------------------------------------- helpers
def _esc(v) -> str:
    return html.escape(str(v), quote=True)


def _pct(v, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.{digits}f} %"


def _num(v, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}".replace(",", " ")


def _prob_bar(p_home: float, p_draw: float, p_away: float) -> str:
    return (
        '<div class="pbar" title="1 / X / 2">'
        f'<span class="pb h" style="width:{p_home*100:.1f}%"></span>'
        f'<span class="pb d" style="width:{p_draw*100:.1f}%"></span>'
        f'<span class="pb a" style="width:{p_away*100:.1f}%"></span>'
        "</div>"
        f'<div class="plabels"><span>{p_home*100:.0f}%</span><span>{p_draw*100:.0f}%</span>'
        f'<span>{p_away*100:.0f}%</span></div>'
    )


def _svg_equity(points: list[float], w: int = 760, h: int = 190, baseline: float | None = None) -> str:
    if len(points) < 2:
        return '<p class="muted">Pas encore assez de données pour tracer la courbe.</p>'
    lo, hi = min(points), max(points)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    span = (hi - lo) or 1.0
    pad = 8
    n = len(points)
    coords = []
    for i, v in enumerate(points):
        x = pad + i * (w - 2 * pad) / (n - 1)
        y = h - pad - (v - lo) / span * (h - 2 * pad)
        coords.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = f"{pad},{h-pad} " + line + f" {w-pad},{h-pad}"
    base_y = None
    if baseline is not None:
        base_y = h - pad - (baseline - lo) / span * (h - 2 * pad)
    base_line = (f'<line x1="{pad}" y1="{base_y:.1f}" x2="{w-pad}" y2="{base_y:.1f}" '
                 'stroke="#475569" stroke-dasharray="4 4" stroke-width="1"/>') if base_y else ""
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" class="chart">'
        f'<defs><linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#34d399" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="#34d399" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        f'<polygon points="{area}" fill="url(#eq)"/>'
        f'{base_line}'
        f'<polyline points="{line}" fill="none" stroke="#34d399" stroke-width="2" stroke-linejoin="round"/>'
        f"</svg>"
        f'<div class="chart-legend"><span>min {_num(lo,0)}</span><span>max {_num(hi,0)}</span></div>'
    )


def _svg_calibration(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">Calibration indisponible.</p>'
    w, h, pad = 520, 190, 26
    bars = []
    n = len(rows)
    bw = (w - 2 * pad) / max(n, 1)
    for i, r in enumerate(rows):
        x = pad + i * bw
        p = float(r.get("predicted", 0))
        o = float(r.get("observed", 0))
        h1 = p * (h - 2 * pad)
        h2 = o * (h - 2 * pad)
        bars.append(
            f'<rect x="{x+2:.1f}" y="{h-pad-h1:.1f}" width="{bw*0.40:.1f}" height="{h1:.1f}" fill="#38bdf8" rx="2"/>'
            f'<rect x="{x+bw*0.46:.1f}" y="{h-pad-h2:.1f}" width="{bw*0.40:.1f}" height="{h2:.1f}" fill="#34d399" rx="2"/>'
            f'<text x="{x+bw*0.34:.1f}" y="{h-pad+13:.1f}" font-size="9" fill="#94a3b8" text-anchor="middle">{_esc(r.get("bin",""))}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" class="chart small">' + "".join(bars) +
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#334155"/></svg>'
        '<div class="chart-legend"><span class="k blue">probabilité annoncée</span>'
        '<span class="k green">fréquence observée</span></div>'
    )


# ---------------------------------------------------------------- sections
def _ou_cell(p: dict) -> str:
    ou = p.get("ou25") or {}
    if not ou:
        return "—"
    sharp = p.get("sharp_totals") or {}
    tail = (f'<br><span class="muted">marché {sharp["over"]:.2f}/{sharp["under"]:.2f}</span>'
            if sharp else "")
    return f'+{ou["fair_over"]:.2f} · −{ou["fair_under"]:.2f}{tail}'


def _ah_cell(p: dict) -> str:
    ah = p.get("ah") or {}
    if not ah:
        return "—"
    line = ah["line"]
    sign = f"{line:+.2f}".rstrip("0").rstrip(".")
    return (f'<span class="muted">dom {sign}</span><br>'
            f'{ah["fair_home"]:.2f} · {ah["fair_away"]:.2f}')


def _kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-l">{_esc(label)}</div>'
            f'<div class="kpi-v">{_esc(value)}</div>'
            f'{f"<div class=kpi-s>{_esc(sub)}</div>" if sub else ""}</div>')


def _bet_card(b: dict) -> str:
    flag = "" if b.get("multi_book") else '<span class="tag warn">source unique</span>'
    return f"""
    <article class="card bet">
      <div class="bet-top">
        <span class="league">{_esc(LEAGUE_LABEL.get(b['league'], b['league']))}</span>
        <span class="tag green">edge {b['edge']*100:+.1f} %</span>{flag}
      </div>
      <h3>{_esc(b['home'])} <span class="muted">vs</span> {_esc(b['away'])}</h3>
      <div class="bet-line">
        <div><span class="muted">Pari</span><b>{_esc(b['label'])}</b></div>
        <div><span class="muted">Cote</span><b>{b['price']:.2f}</b></div>
        <div><span class="muted">Probabilité juste</span><b>{b['p_fair']*100:.1f} %</b></div>
        <div><span class="muted">Mise</span><b>{b['stake']:.2f} u</b></div>
      </div>
      <div class="bet-foot muted">{_esc(b['kickoff'])} · {_esc(b.get('book','—'))} · marché sharp
        {b['p_sharp']*100:.1f} %</div>
    </article>"""


STATUS_FR = {"won": ("gagné", "green"), "lost": ("perdu", "red"),
             "push": ("remboursé", "blue"), "half_won": ("demi-gain", "green"),
             "half_lost": ("demi-perte", "warn")}


def _pnl_span(v: float) -> str:
    cls = "pos" if v > 0 else ("neg" if v < 0 else "neutral")
    return f'<span class="{cls}">{v:+.2f} u</span>'


def _result_row(b: dict) -> str:
    label, tone = STATUS_FR.get(b.get("status", ""), ("—", "warn"))
    return (
        f'<tr><td class="lg">{_esc(str(b.get("kickoff", ""))[:10])}</td>'
        f'<td class="teams"><b>{_esc(b["home"])}</b><span class="muted"> — </span>{_esc(b["away"])}</td>'
        f'<td class="score">{_esc(b.get("result", "—"))}</td>'
        f'<td>{_esc(b.get("label", b.get("selection", "")))}</td>'
        f'<td class="num">{float(b.get("price", 0)):.2f}</td>'
        f'<td class="num">{float(b.get("stake", 0)):.2f} u</td>'
        f'<td class="num">{_pnl_span(float(b.get("pnl") or 0))}</td>'
        f'<td><span class="tag {tone}">{label}</span></td></tr>')


def _hist_row(e: dict) -> str:
    p = e["p"]
    outcome_fr = {"home": "1", "draw": "X", "away": "2"}.get(e.get("outcome"), "—")
    pick = max(p, key=p.get)
    ok = bool(e.get("hit"))
    tag = ('<span class="tag green">bon</span>' if ok
           else '<span class="tag red">manqué</span>')
    return (
        f'<tr><td class="lg">{_esc(e["kickoff"])}</td>'
        f'<td class="teams"><b>{_esc(e["home"])}</b><span class="muted"> — </span>{_esc(e["away"])}</td>'
        f'<td class="num">{p["home"]*100:.0f} / {p["draw"]*100:.0f} / {p["away"]*100:.0f}</td>'
        f'<td class="score">{_esc(e.get("result", ""))}</td>'
        f'<td>{outcome_fr}</td><td>{tag} <span class="muted">{SEL_FR.get(pick, "")}</span></td></tr>')


def _pred_row(p: dict) -> str:
    pr = p["p"]
    value_tag = ""
    if p.get("best_odds"):
        best = p["best_odds"]
        if pr["home"] * best.get("home", 0) - 1 > 0.03:
            value_tag = '<span class="tag green">1</span>'
        elif pr["away"] * best.get("away", 0) - 1 > 0.03:
            value_tag = '<span class="tag green">2</span>'
    sharp = p.get("sharp_odds") or {}
    return f"""
      <tr>
        <td class="lg">{_esc(LEAGUE_LABEL.get(p['league'], p['league']))}</td>
        <td class="teams"><b>{_esc(p['home'])}</b><span class="muted"> — </span>{_esc(p['away'])}</td>
        <td class="probcell">{_prob_bar(pr['home'], pr['draw'], pr['away'])}</td>
        <td class="num">{p['fair']['home']:.2f} · {p['fair']['draw']:.2f} · {p['fair']['away']:.2f}</td>
        <td class="num">{f"{sharp.get('home',0):.2f} · {sharp.get('draw',0):.2f} · {sharp.get('away',0):.2f}" if sharp else "—"}</td>
        <td class="num">{_ou_cell(p)}</td>
        <td class="num">{_ah_cell(p)}</td>
        <td class="num">{p['xg']['home']:.2f} - {p['xg']['away']:.2f}</td>
        <td class="num">{_esc(p['scores'][0][0]) if p.get('scores') else "—"}</td>
        <td>{value_tag}</td>
      </tr>"""


def render(cfg, site: Path | None = None) -> Path:
    site = site or cfg.site_dir
    data_dir = site / "data"

    def load(name: str, default):
        p = data_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return default
        return default

    preds = load("predictions.json", {})
    ledger = load("ledger.json", {})
    backtest = load("backtest.json", {})
    ratings = load("ratings.json", {})
    history = load("history.json", {})
    stats = ledger.get("stats", {})
    fixtures = preds.get("fixtures", [])
    bets = preds.get("bets", [])
    model = preds.get("model", {})
    perf = backtest.get("performance", {})
    predm = backtest.get("predictions", {})

    equity = [float(e["bankroll"]) for e in ledger.get("equity", [])][-300:]
    bt_equity = [float(e["bankroll"]) for e in backtest.get("equity_curve", [])]

    # ----- KPI
    kpis = "".join([
        _kpi("Bankroll papier", f"{_num(stats.get('bankroll', 0), 0)} u",
             f"départ {_num(stats.get('initial_bankroll', 1000), 0)} u"),
        _kpi("Paris réglés", f"{stats.get('n_settled', 0)}",
             f"{stats.get('n_open', 0)} en cours"),
        _kpi("ROI (papier)", _pct(stats.get("roi", 0)), tone="accent" if (stats.get("roi") or 0) > 0 else ""),
        _kpi("Taux de réussite", f"{(stats.get('win_rate') or 0)*100:.1f} %"),
        _kpi("CLV moyen", _pct(stats.get("clv_avg")), "battre la clôture"),
        _kpi("Exposition", f"{_num(stats.get('exposure', 0), 2)} u", "mises ouvertes"),
    ])

    # ----- backtest KPI
    bt_kpis = "".join([
        _kpi("Log loss", _num(predm.get("logloss"), 4), f"n = {predm.get('n', 0)} matchs"),
        _kpi("Brier", _num(predm.get("brier"), 4)),
        _kpi("RPS", _num(predm.get("rps"), 4)),
        _kpi("Précision 1X2", f"{(predm.get('accuracy') or 0)*100:.1f} %"),
        _kpi("ROI backtest", _pct(perf.get("roi")), f"{perf.get('n_bets', 0)} paris"),
        _kpi("Drawdown max", _pct(-(perf.get("max_drawdown") or 0)), tone=""),
    ])

    # ----- résultats réglés (les plus récents d'abord)
    all_bets = ledger.get("bets", [])
    settled_bets = [b for b in all_bets if b.get("status") in ("won", "lost", "push",
                                                               "half_won", "half_lost")]
    settled_bets.sort(key=lambda b: (str(b.get("settled_at") or b.get("kickoff") or ""),
                                     str(b.get("bet_id"))), reverse=True)
    results_rows = "".join(_result_row(b) for b in settled_bets[:25]) or (
        '<tr><td colspan="8" class="muted">Aucun pari réglé pour le moment — les premiers '
        'résultats apparaîtront dès la fin des matchs engagés.</td></tr>')
    last7 = [b for b in settled_bets if str(b.get("settled_at", ""))[:10] >=
             (datetime.now(UTC).date() - timedelta(days=7)).isoformat()]
    pnl7 = sum(float(b.get("pnl") or 0) for b in last7)
    n_win7 = sum(1 for b in last7 if b.get("status") in ("won", "half_won"))
    results_kpi = "".join([
        _kpi("Paris réglés", str(len(settled_bets)), f"{len(last7)} sur 7 jours"),
        _kpi("P&L 7 jours", f"{pnl7:+.2f} u", tone="accent" if pnl7 > 0 else ""),
        _kpi("Gagnants 7 j", f"{n_win7}/{len(last7)}" if last7 else "—"),
        _kpi("Meilleur pari", f"{max((float(b.get('pnl') or 0) for b in settled_bets), default=0):+.2f} u"),
        _kpi("Pire pari", f"{min((float(b.get('pnl') or 0) for b in settled_bets), default=0):+.2f} u"),
    ])

    # ----- historique des pronostics
    hstats = history.get("stats", {}) or {}
    hstats30 = history.get("stats_30d", {}) or {}
    hist_kpi = "".join([
        _kpi("Pronostics vérifiés", str(hstats.get("n", 0)),
             f"{hstats.get('n_total', 0)} archivés au total"),
        _kpi("Taux de réussite", f"{(hstats.get('hit_rate') or 0)*100:.1f} %"
             if hstats.get("n") else "—", "issue la plus probable"),
        _kpi("30 derniers jours", f"{(hstats30.get('hit_rate') or 0)*100:.1f} %"
             if hstats30.get("n") else "—", f"{hstats30.get('n', 0)} matchs"),
        _kpi("Log loss", _num(hstats.get("logloss"), 4),
             "marché ≈ 0,97 — plus bas = mieux"),
        _kpi("Brier", _num(hstats.get("brier"), 4)),
    ])
    hist_rows = "".join(_hist_row(e) for e in history.get("recent", [])[:25]) or (
        '<tr><td colspan="6" class="muted">Les pronostics s\'archivent à chaque exécution ; '
        'ils seront vérifiés dès que les matchs seront joués.</td></tr>')
    hist_league_rows = "".join(
        f'<tr><td>{_esc(LEAGUE_LABEL.get(r["league"], r["league"]))}</td>'
        f'<td class="num">{r["n"]}</td><td class="num">{r["hit_rate"]*100:.1f} %</td></tr>'
        for r in history.get("by_league", [])[:14]) or (
        '<tr><td colspan="3" class="muted">—</td></tr>')

    bet_cards = "".join(_bet_card(b) for b in bets) or (
        '<p class="muted empty">Aucun value bet retenu sur la fenêtre courante — '
        "c'est aussi un résultat : le marché était efficient.</p>")

    rows = "".join(_pred_row(p) for p in fixtures) or (
        '<tr><td colspan="8" class="muted">Aucun match dans la fenêtre publiée.</td></tr>')

    # ----- classements
    rating_blocks = []
    for code in sorted(ratings.keys(), key=lambda c: -len(ratings[c])):
        items = ratings[code][:5]
        if not items:
            continue
        lis = "".join(
            f'<li><span class="rk">{i+1}</span><span class="tn">{_esc(t["team"])}</span>'
            f'<span class="rt">{t["rating"]:+.2f}</span></li>'
            for i, t in enumerate(items))
        rating_blocks.append(
            f'<div class="rating-card"><h4>{_esc(LEAGUE_LABEL.get(code, code))}</h4><ol>{lis}</ol></div>')

    by_league = backtest.get("by_league", [])
    league_rows = "".join(
        f'<tr><td>{_esc(LEAGUE_LABEL.get(r["group"], r["group"]))}</td><td class="num">{r["n"]}</td>'
        f'<td class="num">{_pct(r["roi"])}</td><td class="num">{_num(r["pnl"], 1)}</td></tr>'
        for r in by_league[:12])

    per_season = "".join(
        f'<tr><td>{s["season"]}/{str(s["season"]+1)[-2:]}</td><td class="num">{s["n_bets"]}</td>'
        f'<td class="num">{_pct(s["roi"])}</td><td class="num">{_num(s["pnl"], 1)}</td>'
        f'<td class="num">{s.get("w_dc", "—")}</td></tr>'
        for s in backtest.get("per_season", []))

    ts = preds.get("generated_at", "")
    try:
        ts_fr = datetime.fromisoformat(ts).strftime("%d/%m/%Y %H:%M UTC") if ts else "—"
    except ValueError:
        ts_fr = ts

    tpl = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")
    out = (tpl
           .replace("__GENERATED__", _esc(ts_fr))
           .replace("__VERSION__", _esc(model.get("version", "—")))
           .replace("__MODEL_WDC__", _esc(str(model.get("w_dc", "—"))))
           .replace("__MODEL_T__", _esc(str(model.get("temperature", "—"))))
           .replace("__N_MATCHES__", _esc(str(model.get("n_matches_train", "—"))))
           .replace("__N_FIXTURES__", str(len(fixtures)))
           .replace("__KPI__", kpis)
           .replace("__BT_KPI__", bt_kpis)
           .replace("__BETS__", bet_cards)
           .replace("__ROWS__", rows)
           .replace("__RESULTS_KPI__", results_kpi)
           .replace("__RESULTS_ROWS__", results_rows)
           .replace("__HIST_KPI__", hist_kpi)
           .replace("__HIST_ROWS__", hist_rows)
           .replace("__HIST_LEAGUE_ROWS__", hist_league_rows)
           .replace("__EQUITY__", _svg_equity(bt_equity, baseline=backtest.get("initial_bankroll")))
           .replace("__LEDGER_EQUITY__", _svg_equity(equity, h=120,
                                                     baseline=stats.get("initial_bankroll")))
           .replace("__CALIB__", _svg_calibration(predm.get("calibration", [])))
           .replace("__RATINGS__", "".join(rating_blocks) or '<p class="muted">—</p>')
           .replace("__LEAGUE_TABLE__", league_rows or '<tr><td colspan="4" class="muted">—</td></tr>')
           .replace("__SEASON_TABLE__", per_season or '<tr><td colspan="5" class="muted">—</td></tr>'))

    out_path = site / "index.html"
    out_path.write_text(out, encoding="utf-8")
    log.info("dashboard écrit -> %s (%.0f Ko)", out_path, out_path.stat().st_size / 1024)
    return out_path
