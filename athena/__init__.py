"""ATHENA - moteur quantitatif de pronostics sportifs.

Pipeline : ingestion (sources gratuites) -> features -> modeles
(Dixon-Coles + Elo + GBM) -> calibration -> value betting (Kelly)
-> backtest walk-forward -> publication (dashboard statique).
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
