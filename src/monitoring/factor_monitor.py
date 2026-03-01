"""
Factor monitoring module for AdaptiveCompositeManager.

Tracks three health dimensions in real time and fires loguru alerts
when thresholds are breached:

  1. ICIR (IC Information Ratio) per factor
     — rolling IC mean / IC std over the last ic_window observations
     — logger.warning when ICIR < icir_threshold

  2. Weight Herfindahl-Hirschman Index (HHI)
     — sum of squared weights; near 1/N is healthy, near 1 is degenerate
     — logger.warning when HHI > hhi_threshold

  3. IC skip staleness
     — reads _ic_skip_count from the manager (populated by Issue #2)
     — logger.error when any factor has >= forward_period consecutive skips

Optionally persists all metric snapshots to a SQLite database.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import IC_SIGNIFICANCE_THRESHOLD


class FactorMonitor:
    """
    Real-time health monitor for an AdaptiveCompositeManager.

    Usage
    -----
    >>> monitor = FactorMonitor(manager)
    >>> for date in dates:
    ...     composite, weights = manager.compute_factor_with_update(data, prices, date)
    ...     metrics = monitor.check(date)
    >>> monitor.summary()

    Parameters
    ----------
    manager : AdaptiveCompositeManager
        The manager instance to monitor.
    icir_threshold : float
        Minimum acceptable rolling ICIR. Default 0.3 (Grinold & Kahn standard).
        Uses the same ic_window as the manager for the rolling computation.
    hhi_threshold : float
        Maximum acceptable weight HHI. Default 0.5.
        In a 3-factor system, equal weights give HHI ≈ 0.33;
        a value of 0.5 means one factor is strongly dominant.
    db_path : str, optional
        Path to a SQLite file for cross-session metric persistence.
        If None (default), metrics are kept in memory only.
    """

    def __init__(
        self,
        manager,
        icir_threshold: float = 0.3,
        hhi_threshold: float = 0.5,
        db_path: Optional[str] = None,
    ):
        self.manager = manager
        self.icir_threshold = icir_threshold
        self.hhi_threshold = hhi_threshold
        self.db_path = db_path

        self._metrics_history: List[Dict] = []
        self._active_alerts: List[str] = []

        if db_path is not None:
            self._init_db()

        factor_names = [f.name for f in manager.adaptive_factor.factors]
        logger.info(f"FactorMonitor initialized: {len(factor_names)} factors — {factor_names}")
        logger.info(f"  icir_threshold={icir_threshold}, hhi_threshold={hhi_threshold}")
        if db_path:
            logger.info(f"  Persisting metrics to: {db_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, current_date=None) -> Dict:
        """
        Run all health checks and return a metrics snapshot.

        Call this after each AdaptiveCompositeManager.compute_factor_with_update().

        Parameters
        ----------
        current_date : pd.Timestamp or str, optional
            Date to associate with this snapshot. Defaults to now.

        Returns
        -------
        dict with keys:
            date        : the snapshot date
            icir        : {factor_name: float}  rolling ICIR per factor
            ic_mean     : {factor_name: float}  rolling IC mean per factor
            hhi         : float                 weight HHI
            skip_counts : {factor_name: int}    consecutive IC skip count
            alerts      : [str]                 alert messages fired this check
        """
        if current_date is None:
            current_date = datetime.now()

        self._active_alerts = []

        icir_metrics = self._check_icir()
        ic_mean_metrics = self._get_ic_means()
        hhi = self._check_hhi()
        skip_counts = dict(self.manager._ic_skip_count)

        # Surface stale-weight errors for external consumers
        fwd = self.manager.forward_period
        for factor_name, count in skip_counts.items():
            if count >= fwd:
                msg = (
                    f"Factor {factor_name} IC has not updated for "
                    f"{count} consecutive periods — weights are stale"
                )
                logger.error(msg)
                self._active_alerts.append(msg)

        metrics = {
            "date": current_date,
            "icir": icir_metrics,
            "ic_mean": ic_mean_metrics,
            "hhi": hhi,
            "skip_counts": skip_counts,
            "alerts": list(self._active_alerts),
        }

        self._metrics_history.append(metrics)

        if self.db_path is not None:
            self._persist(metrics)

        return metrics

    def summary(self) -> None:
        """Log a human-readable health summary of all monitored factors."""
        af = self.manager.adaptive_factor
        ic_summary = af.get_ic_summary()
        weight_df = af.get_weight_history()

        logger.info("=" * 60)
        logger.info("FactorMonitor — Health Summary")
        logger.info("=" * 60)

        # IC statistics
        if ic_summary.empty:
            logger.info("  IC: no history yet (still in warm-up period)")
        else:
            logger.info("  IC Statistics (all history):")
            for factor_name, row in ic_summary.iterrows():
                ic_ir = row.get("ic_ir", float("nan"))
                flag = " [!]" if (not np.isnan(ic_ir) and ic_ir < self.icir_threshold) else ""
                logger.info(
                    f"    {factor_name}: "
                    f"mean={row['ic_mean']:.4f}, "
                    f"std={row['ic_std']:.4f}, "
                    f"ICIR={ic_ir:.3f}{flag}, "
                    f"positive={row['ic_positive_rate']:.1%}, "
                    f"n={int(row['n_observations'])}"
                )

        # Latest weights and HHI
        if not weight_df.empty:
            latest_weights = weight_df.iloc[-1]
            n = len(latest_weights)
            hhi = float((latest_weights ** 2).sum())
            hhi_flag = " [!]" if hhi > self.hhi_threshold else ""
            dominant = latest_weights.idxmax()
            logger.info(
                f"  Weights: HHI={hhi:.3f}{hhi_flag} "
                f"(equal={1/n:.3f}, dominant={dominant})"
            )
            for factor_name, w in latest_weights.items():
                logger.info(f"    {factor_name}: {w:.3f} ({w * 100:.1f}%)")

        # IC skip counts
        skip = self.manager._ic_skip_count
        if skip:
            logger.info("  IC skip counts (consecutive):")
            fwd = self.manager.forward_period
            for name, count in skip.items():
                flag = " [! STALE]" if count >= fwd else ""
                logger.info(f"    {name}: {count}{flag}")

        # Active alerts
        if self._active_alerts:
            logger.info(f"  Active alerts ({len(self._active_alerts)}):")
            for alert in self._active_alerts:
                logger.info(f"    - {alert}")
        else:
            logger.info("  No active alerts")

        logger.info("=" * 60)

    @property
    def alerts(self) -> List[str]:
        """Alert messages from the most recent check()."""
        return list(self._active_alerts)

    def get_metrics_history(self) -> pd.DataFrame:
        """
        Return all collected metric snapshots as a flat DataFrame.

        Columns: date, factor_name, icir, ic_mean, hhi, skip_count

        If db_path was provided, also includes snapshots from previous sessions
        loaded from SQLite.
        """
        if self.db_path is not None:
            return self._load_from_db()

        if not self._metrics_history:
            return pd.DataFrame()

        return self._snapshots_to_df(self._metrics_history)

    # ------------------------------------------------------------------
    # Private helpers — checks
    # ------------------------------------------------------------------

    def _check_icir(self) -> Dict[str, float]:
        """Compute rolling ICIR per factor. Emit warning when below threshold."""
        af = self.manager.adaptive_factor
        ic_window = af.ic_window
        result = {}

        for factor_name, ic_list in af.ic_history.items():
            if len(ic_list) < 2:
                result[factor_name] = float("nan")
                continue

            recent = ic_list[-ic_window:]
            ic_values = np.array([x[1] for x in recent])
            ic_std = ic_values.std()

            if ic_std < 1e-9:
                icir = float("nan")
            else:
                icir = float(ic_values.mean() / ic_std)

            result[factor_name] = icir

            if not np.isnan(icir) and icir < self.icir_threshold:
                msg = (
                    f"Factor {factor_name} ICIR={icir:.3f} is below threshold "
                    f"{self.icir_threshold} (n={len(recent)})"
                )
                logger.warning(msg)
                self._active_alerts.append(msg)

        return result

    def _get_ic_means(self) -> Dict[str, float]:
        """Return rolling IC mean per factor (last ic_window entries)."""
        af = self.manager.adaptive_factor
        ic_window = af.ic_window
        result = {}

        for factor_name, ic_list in af.ic_history.items():
            if not ic_list:
                result[factor_name] = float("nan")
                continue
            recent = ic_list[-ic_window:]
            ic_values = np.array([x[1] for x in recent])
            result[factor_name] = float(ic_values.mean())

        return result

    def _check_hhi(self) -> float:
        """Compute HHI from the latest weight snapshot. Emit warning if above threshold."""
        weight_history = self.manager.adaptive_factor.weight_history
        if not weight_history:
            return float("nan")

        weights = weight_history[-1]["weights"]
        w_array = np.array(list(weights.values()))
        hhi = float((w_array ** 2).sum())

        n = len(w_array)
        if hhi > self.hhi_threshold:
            dominant = max(weights, key=weights.get)
            msg = (
                f"Weight concentration HHI={hhi:.3f} exceeds threshold {self.hhi_threshold} "
                f"(equal={1/n:.3f}, dominant factor: {dominant}={weights[dominant]:.3f})"
            )
            logger.warning(msg)
            self._active_alerts.append(msg)

        return hhi

    # ------------------------------------------------------------------
    # Private helpers — SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the metrics table if it doesn't already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_metrics (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    date        TEXT,
                    factor_name TEXT NOT NULL,
                    icir        REAL,
                    ic_mean     REAL,
                    hhi         REAL,
                    skip_count  INTEGER
                )
            """)
            conn.commit()
        logger.info(f"FactorMonitor: SQLite initialized at {self.db_path}")

    def _persist(self, metrics: Dict) -> None:
        """Write one snapshot to SQLite."""
        recorded_at = datetime.now().isoformat()
        date_str = str(metrics["date"])
        hhi = metrics["hhi"]
        hhi_val = None if (isinstance(hhi, float) and np.isnan(hhi)) else hhi

        rows = [
            (
                recorded_at,
                date_str,
                factor_name,
                metrics["icir"].get(factor_name),
                metrics["ic_mean"].get(factor_name),
                hhi_val,
                metrics["skip_counts"].get(factor_name, 0),
            )
            for factor_name in metrics["icir"]
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO factor_metrics
                   (recorded_at, date, factor_name, icir, ic_mean, hhi, skip_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

    def _load_from_db(self) -> pd.DataFrame:
        """Load all rows from SQLite into a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT date, factor_name, icir, ic_mean, hhi, skip_count "
                "FROM factor_metrics ORDER BY recorded_at",
                conn,
            )

    @staticmethod
    def _snapshots_to_df(snapshots: List[Dict]) -> pd.DataFrame:
        """Flatten in-memory snapshot list to a DataFrame."""
        rows = []
        for snap in snapshots:
            date = snap["date"]
            hhi = snap["hhi"]
            for factor_name in snap["icir"]:
                rows.append({
                    "date": date,
                    "factor_name": factor_name,
                    "icir": snap["icir"].get(factor_name),
                    "ic_mean": snap["ic_mean"].get(factor_name),
                    "hhi": hhi,
                    "skip_count": snap["skip_counts"].get(factor_name, 0),
                })
        return pd.DataFrame(rows)
