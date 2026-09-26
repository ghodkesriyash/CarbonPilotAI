"""
CarbonPilot AI — Energy Source Switch Planner & Carbon-Water Impact Reporter
=============================================================
Single-file backend. Every model described in Table 3 of the technical
report lives in this file as its own clearly-marked section, orchestrated
by one pipeline function and served over one FastAPI app.

    MODEL 1  LoadForecastModel        -> Fourier-feature regression (Prophet/SARIMA-lite)
    MODEL 2  SolarForecastModel       -> weather-feature regression (cloud cover / irradiance)
    MODEL 3  SourceOptimizer          -> linear program over grid/solar/battery/diesel (PuLP)
    MODEL 4  ImpactCalculator         -> deterministic CO2 + water footprint engine
    MODEL 5  ESGNarrativeGenerator    -> LLM (Claude) narrative writer, template fallback

Run:
    pip install -r requirements.txt
    python app.py
    # -> http://localhost:8000  (docs at /docs)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover
    PULP_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:  # pragma: no cover
    APSCHEDULER_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    REPORTLAB_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANTHROPIC_SDK_AVAILABLE = False


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.jsonl"
LATEST_FILE = DATA_DIR / "latest.json"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Emission / water factors. Sources noted inline; these are published
# reference figures used for a deterministic (non-ML) calculation, exactly
# as scoped in the report ("not ML, but essential to the pipeline").
GRID_EF_BASE_G_PER_KWH = 713.0       # CEA CO2 Baseline Database, India grid average, order-of-magnitude
DIESEL_EF_KG_PER_L = 2.68            # IPCC default emission factor, diesel combustion
DIESEL_GENSET_KWH_PER_L = 3.0        # typical diesel genset output per litre (~0.33 L/kWh)
COOLING_WUE_L_PER_KWH = 1.8          # The Green Grid WUE benchmark, evaporative cooling
GRID_UPSTREAM_WATER_L_PER_KWH = 2.0  # thermal generation cooling-water withdrawal, industry order-of-magnitude
DIESEL_WATER_L_PER_KWH = 0.05        # radiator top-up, small vs. thermal grid generation

DEFAULT_INTERVAL_SECONDS = int(os.environ.get("DEMO_INTERVAL_SECONDS", "3600"))


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class PlanRequest(BaseModel):
    city: str = Field(default="Pune, India")
    lat: float = Field(default=18.5204)
    lon: float = Field(default=73.8567)
    hours: int = Field(default=24, ge=6, le=48)
    dc_capacity_kw: float = Field(default=1000.0, gt=0)
    solar_capacity_kw: float = Field(default=400.0, ge=0)
    battery_capacity_kwh: float = Field(default=500.0, ge=0)
    battery_max_rate_kw: float = Field(default=150.0, ge=0)
    diesel_max_kw: float = Field(default=600.0, ge=0)
    grid_price_per_kwh: float = Field(default=8.0, gt=0)
    diesel_price_per_l: float = Field(default=95.0, gt=0)
    carbon_priority: float = Field(
        default=0.5, ge=0, le=1,
        description="0 = pure lowest-cost, 1 = pure lowest-carbon. Sets the LP's shadow carbon price."
    )
    simulate_outage: bool = Field(default=False)
    outage_start_hour: int = Field(default=14)
    outage_end_hour: int = Field(default=17)


class HourlyRecord(BaseModel):
    hour_index: int
    timestamp: str
    load_kw: float
    solar_available_kw: float
    grid_carbon_intensity_g_per_kwh: float
    grid_price_per_kwh: float
    grid_kw: float
    solar_kw: float
    battery_discharge_kw: float
    battery_charge_kw: float
    diesel_kw: float
    battery_soc_kwh: float
    cost_inr: float
    co2_kg: float
    water_l: float
    grid_outage: bool


class PlanResponse(BaseModel):
    generated_at: str
    city: str
    lat: float
    lon: float
    weather_source: str
    optimizer_engine: str
    hourly: list[HourlyRecord]
    totals: dict
    baseline_totals: dict
    savings: dict
    mix_pct: dict


class ESGNarrativeResponse(BaseModel):
    narrative: str
    generated_with: str


# ============================================================================
# LOCAL STORE  (JSON-file stand-in for Firebase/MongoDB Atlas — same
# schema-flexible document pattern, zero external service needed for a demo)
# ============================================================================

class LocalStore:
    def __init__(self):
        self._lock = threading.Lock()

    def save_plan(self, plan: dict) -> None:
        with self._lock:
            LATEST_FILE.write_text(json.dumps(plan))
            with HISTORY_FILE.open("a") as f:
                f.write(json.dumps({
                    "generated_at": plan["generated_at"],
                    "city": plan["city"],
                    "totals": plan["totals"],
                    "baseline_totals": plan["baseline_totals"],
                    "savings": plan["savings"],
                }) + "\n")

    def load_latest(self) -> Optional[dict]:
        if not LATEST_FILE.exists():
            return None
        return json.loads(LATEST_FILE.read_text())

    def load_history(self, limit: int = 30) -> list[dict]:
        if not HISTORY_FILE.exists():
            return []
        lines = HISTORY_FILE.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-limit:]]


store = LocalStore()
_last_params: Optional[PlanRequest] = None


# ============================================================================
# EXTERNAL DATA FEEDS  (Stage 1 of the report's system workflow)
# ============================================================================

def fetch_weather_forecast(lat: float, lon: float, hours: int) -> tuple[pd.DataFrame, str]:
    """Pull hourly cloud cover + shortwave radiation from Open-Meteo (free,
    no API key). Falls back to a synthetic clear-sky model if the network
    call fails, so the pipeline always produces a demoable result."""
    if httpx is not None:
        try:
            resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "cloudcover,shortwave_radiation,temperature_2m",
                    "forecast_days": 2,
                    "timezone": "auto",
                },
                timeout=6.0,
            )
            resp.raise_for_status()
            data = resp.json()["hourly"]
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["time"]),
                "cloud_cover_pct": data["cloudcover"],
                "shortwave_radiation": data["shortwave_radiation"],
                "temperature_c": data["temperature_2m"],
            })
            now = datetime.now()
            df = df[df["timestamp"] >= now - timedelta(hours=1)].reset_index(drop=True)
            if len(df) >= hours:
                return df.iloc[:hours].reset_index(drop=True), "open-meteo.com (live)"
        except Exception:
            pass  # fall through to synthetic feed

    return _synthetic_weather(hours), "synthetic clear-sky model (offline fallback)"


def _synthetic_weather(hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    rows = []
    cloud_regime = rng.uniform(10, 40)  # slowly-drifting cloud base for the window
    for h in range(hours):
        ts = start + timedelta(hours=h)
        hour_of_day = ts.hour
        # bell-shaped clear-sky irradiance, zero before 6am / after 18:30
        daylight = max(0.0, math.sin(math.pi * (hour_of_day - 6) / 13)) if 6 <= hour_of_day <= 19 else 0.0
        ghi = 950 * daylight
        cloud_regime += rng.normal(0, 4)
        cloud_regime = min(max(cloud_regime, 5), 85)
        rows.append({
            "timestamp": ts,
            "cloud_cover_pct": cloud_regime,
            "shortwave_radiation": ghi,
            "temperature_c": 24 + 6 * daylight + rng.normal(0, 0.5),
        })
    return pd.DataFrame(rows)


def generate_synthetic_load_history(capacity_kw: float, days: int = 21, seed: int = 42) -> pd.DataFrame:
    """A data-centre load profile is much flatter than an office building's:
    it never goes to zero, but still tracks business hours and weekly
    rhythm as batch jobs / user traffic ebb and flow. Used only to *train*
    the forecasting model — clearly a labelled synthetic substitute for
    real DC telemetry, per the report's data-sources section."""
    rng = np.random.default_rng(seed)
    hours = days * 24
    t = np.arange(hours)
    hour_of_day = t % 24
    day_of_week = (t // 24) % 7

    base = 0.80  # DC floor load never drops much — always-on servers
    diurnal_swing = 0.20 * np.sin((hour_of_day - 8) / 24 * 2 * np.pi) ** 2
    diurnal_swing *= np.where((hour_of_day >= 8) & (hour_of_day <= 22), 1.0, 0.3)
    weekend_dip = np.where(day_of_week >= 5, -0.04, 0.0)
    noise = rng.normal(0, 0.018, size=hours)

    load_fraction = base + diurnal_swing + weekend_dip + noise
    load_fraction = np.clip(load_fraction, 0.55, 1.05)
    load_kw = load_fraction * capacity_kw

    start = datetime.now() - timedelta(hours=hours)
    timestamps = [start + timedelta(hours=int(h)) for h in t]
    return pd.DataFrame({"timestamp": timestamps, "hour_index": t, "load_kw": load_kw})


# ============================================================================
# MODEL 1 — LOAD FORECASTING
# Report spec: "Facebook Prophet or a simple SARIMA model ... both handle
# daily/weekly seasonality with minimal tuning." Implemented here as a
# Fourier-feature linear regression: same seasonality-capture idea, zero
# heavy dependencies, trains in milliseconds — a deliberate lightweight
# stand-in that is swappable for Prophet/SARIMA/LSTM later (Table 3,
# "stretch / future upgrade").
# ============================================================================

class LoadForecastModel:
    N_HARMONICS_DAILY = 3
    N_HARMONICS_WEEKLY = 2

    def __init__(self):
        self.model = LinearRegression()

    def _features(self, t: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(t, dtype=float)]
        for k in range(1, self.N_HARMONICS_DAILY + 1):
            cols.append(np.sin(2 * np.pi * k * t / 24))
            cols.append(np.cos(2 * np.pi * k * t / 24))
        for k in range(1, self.N_HARMONICS_WEEKLY + 1):
            cols.append(np.sin(2 * np.pi * k * t / (24 * 7)))
            cols.append(np.cos(2 * np.pi * k * t / (24 * 7)))
        return np.column_stack(cols)

    def fit_and_predict(self, capacity_kw: float, hours: int) -> pd.DataFrame:
        history = generate_synthetic_load_history(capacity_kw)
        t_hist = history["hour_index"].to_numpy()
        X_hist = self._features(t_hist)
        y_hist = history["load_kw"].to_numpy()
        self.model.fit(X_hist, y_hist)

        t_future = np.arange(t_hist[-1] + 1, t_hist[-1] + 1 + hours)
        X_future = self._features(t_future)
        forecast = self.model.predict(X_future)
        forecast = np.clip(forecast, 0.5 * capacity_kw, 1.1 * capacity_kw)

        start = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        timestamps = [start + timedelta(hours=h) for h in range(hours)]
        return pd.DataFrame({"timestamp": timestamps, "load_kw": forecast})


# ============================================================================
# MODEL 2 — RENEWABLE (SOLAR) AVAILABILITY FORECASTING
# Report spec: "Regression model (linear regression or XGBoost) using
# weather-API features such as cloud cover and irradiance proxy."
# Implemented as a scikit-learn LinearRegression trained on a small
# physics-motivated synthetic dataset, then applied to real (or
# fallback-synthetic) weather features for the forecast window.
# ============================================================================

class SolarForecastModel:
    def __init__(self, panel_performance_ratio: float = 0.80):
        self.performance_ratio = panel_performance_ratio
        self.model = LinearRegression()
        self._train()

    def _train(self):
        rng = np.random.default_rng(11)
        n = 500
        ghi_norm = rng.uniform(0, 1, n)          # shortwave radiation / 1000 W/m^2
        cloud = rng.uniform(0, 1, n)              # cloud cover fraction
        output_fraction = np.clip(
            ghi_norm * (1 - 0.75 * cloud) + rng.normal(0, 0.02, n), 0, 1
        )
        X = np.column_stack([ghi_norm, cloud])
        self.model.fit(X, output_fraction)

    def predict(self, weather: pd.DataFrame, solar_capacity_kw: float) -> np.ndarray:
        ghi_norm = np.clip(weather["shortwave_radiation"].to_numpy() / 1000.0, 0, 1.3)
        cloud = np.clip(weather["cloud_cover_pct"].to_numpy() / 100.0, 0, 1)
        X = np.column_stack([ghi_norm, cloud])
        output_fraction = np.clip(self.model.predict(X), 0, 1)
        return output_fraction * solar_capacity_kw * self.performance_ratio


# ============================================================================
# MODEL 3 — SOURCE OPTIMISATION ENGINE
# Report spec: "Rule-based multi-objective optimisation using SciPy or
# PuLP (linear programming) over forecasted load, availability, and carbon
# intensity." Implemented as a genuine linear program (PuLP + CBC) coupling
# all hours of the horizon through the battery's state of charge, with a
# tunable carbon shadow-price so the same engine can be pushed from
# cost-optimal to carbon-optimal (the "carbon_priority" slider on the
# dashboard). Falls back to a simple greedy rule-based allocator if PuLP is
# unavailable, matching the report's named fallback approach.
# ============================================================================

def grid_carbon_intensity_series(timestamps: list[datetime]) -> np.ndarray:
    """More coal-heavy baseload at night, cleaner mix midday when solar
    feeds the wider grid — a simplified diurnal curve around the CEA
    baseline average."""
    hours = np.array([ts.hour for ts in timestamps])
    return GRID_EF_BASE_G_PER_KWH * (1 + 0.14 * np.cos(2 * np.pi * (hours - 3) / 24))


def grid_tariff_series(timestamps: list[datetime], base_price: float) -> np.ndarray:
    """Simple time-of-use tariff: peak / off-peak / normal bands."""
    hours = np.array([ts.hour for ts in timestamps])
    price = np.full(len(hours), base_price)
    peak = ((hours >= 9) & (hours < 12)) | ((hours >= 18) & (hours < 22))
    offpeak = (hours >= 22) | (hours < 6)
    price[peak] = base_price * 1.3
    price[offpeak] = base_price * 0.8
    return price


def optimize_sources(
    load_kw: np.ndarray,
    solar_avail_kw: np.ndarray,
    grid_ef: np.ndarray,
    grid_price: np.ndarray,
    timestamps: list[datetime],
    req: PlanRequest,
) -> tuple[pd.DataFrame, str]:
    H = len(load_kw)
    outage = np.zeros(H, dtype=bool)
    if req.simulate_outage:
        for i, ts in enumerate(timestamps):
            if req.outage_start_hour <= ts.hour < req.outage_end_hour:
                outage[i] = True

    diesel_price_per_kwh = req.diesel_price_per_l / DIESEL_GENSET_KWH_PER_L
    diesel_ef_per_kwh = DIESEL_EF_KG_PER_L / DIESEL_GENSET_KWH_PER_L  # kg CO2 per kWh generated
    # Shadow carbon price: at priority=0 -> ~0 INR/kg (pure cost); at
    # priority=1 -> a strong ~35 INR/kg that dominates the objective.
    carbon_price_per_kg = 35.0 * (req.carbon_priority ** 1.5)

    battery_eff = 0.9
    soc_min = 0.10 * req.battery_capacity_kwh
    soc_max = 0.95 * req.battery_capacity_kwh
    soc_init = 0.5 * req.battery_capacity_kwh

    if PULP_AVAILABLE and req.battery_capacity_kwh >= 0:
        try:
            prob = pulp.LpProblem("dc_source_plan", pulp.LpMinimize)
            idx = range(H)
            grid = pulp.LpVariable.dicts("grid", idx, lowBound=0)
            solar = pulp.LpVariable.dicts("solar", idx, lowBound=0)
            batt_c = pulp.LpVariable.dicts("batt_charge", idx, lowBound=0, upBound=req.battery_max_rate_kw)
            batt_d = pulp.LpVariable.dicts("batt_discharge", idx, lowBound=0, upBound=req.battery_max_rate_kw)
            diesel = pulp.LpVariable.dicts("diesel", idx, lowBound=0, upBound=req.diesel_max_kw)
            soc = pulp.LpVariable.dicts("soc", range(H + 1), lowBound=soc_min, upBound=soc_max)

            prob += soc[0] == soc_init
            for h in idx:
                prob += solar[h] <= float(solar_avail_kw[h])
                if outage[h]:
                    prob += grid[h] == 0
                else:
                    prob += diesel[h] == 0  # diesel is backup-only, reserved for outage hours
                prob += (solar[h] + grid[h] + batt_d[h] + diesel[h] - batt_c[h]) == float(load_kw[h])
                prob += soc[h + 1] == soc[h] + battery_eff * batt_c[h] - batt_d[h] * (1.0 / battery_eff)

            cost_term = pulp.lpSum(
                grid_price[h] * grid[h] + diesel_price_per_kwh * diesel[h] for h in idx
            )
            carbon_term = pulp.lpSum(
                carbon_price_per_kg * (
                    (grid_ef[h] / 1000.0) * grid[h] + diesel_ef_per_kwh * diesel[h]
                ) for h in idx
            )
            prob += cost_term + carbon_term

            solver = pulp.PULP_CBC_CMD(msg=False)
            prob.solve(solver)

            if pulp.LpStatus[prob.status] == "Optimal":
                rows = []
                for h in idx:
                    rows.append({
                        "grid_kw": max(0.0, grid[h].value() or 0.0),
                        "solar_kw": max(0.0, solar[h].value() or 0.0),
                        "battery_discharge_kw": max(0.0, batt_d[h].value() or 0.0),
                        "battery_charge_kw": max(0.0, batt_c[h].value() or 0.0),
                        "diesel_kw": max(0.0, diesel[h].value() or 0.0),
                        "battery_soc_kwh": max(0.0, soc[h + 1].value() or 0.0),
                        "grid_outage": bool(outage[h]),
                    })
                return pd.DataFrame(rows), "linear program (PuLP / CBC)"
        except Exception:
            pass  # fall through to greedy allocator

    return _greedy_allocate(load_kw, solar_avail_kw, grid_ef, grid_price, outage, req), "rule-based greedy allocator (fallback)"


def _greedy_allocate(load_kw, solar_avail_kw, grid_ef, grid_price, outage, req: PlanRequest) -> pd.DataFrame:
    """Fallback hour-by-hour rule-based allocator: solar first, then battery
    when it's cheap/clean to do so, then grid, then diesel only on outage."""
    H = len(load_kw)
    soc = 0.5 * req.battery_capacity_kwh
    soc_min = 0.10 * req.battery_capacity_kwh
    soc_max = 0.95 * req.battery_capacity_kwh
    battery_eff = 0.9
    price_median = float(np.median(grid_price))
    ef_median = float(np.median(grid_ef))
    rows = []
    for h in range(H):
        remaining = float(load_kw[h])
        solar_use = min(remaining, float(solar_avail_kw[h]))
        remaining -= solar_use
        batt_discharge = 0.0
        batt_charge = 0.0
        diesel_use = 0.0
        grid_use = 0.0

        expensive_or_dirty = (grid_price[h] > price_median) or (grid_ef[h] > ef_median)
        if remaining > 0 and not outage[h] and expensive_or_dirty and soc > soc_min:
            batt_discharge = min(remaining, req.battery_max_rate_kw, (soc - soc_min) * battery_eff)
            remaining -= batt_discharge
            soc -= batt_discharge / battery_eff

        if outage[h]:
            diesel_use = min(remaining, req.diesel_max_kw)
            remaining -= diesel_use
        else:
            grid_use = remaining
            remaining = 0.0

        surplus_solar = max(0.0, float(solar_avail_kw[h]) - solar_use)
        if surplus_solar > 0 and soc < soc_max:
            batt_charge = min(surplus_solar, req.battery_max_rate_kw, (soc_max - soc) / battery_eff)
            soc += batt_charge * battery_eff

        rows.append({
            "grid_kw": grid_use, "solar_kw": solar_use,
            "battery_discharge_kw": batt_discharge, "battery_charge_kw": batt_charge,
            "diesel_kw": diesel_use, "battery_soc_kwh": soc, "grid_outage": bool(outage[h]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# MODEL 4 — CARBON & WATER IMPACT CALCULATOR
# Report spec: deterministic calculation using published IPCC/CPCB emission
# factors and industry WUE benchmarks. Not ML — but essential to the
# pipeline, and reused to build a "business as usual" baseline for savings.
# ============================================================================

def compute_impact_row(row: pd.Series, grid_price: float, diesel_price_per_kwh: float) -> dict:
    co2_kg = (
        row["grid_kw"] * (row["grid_carbon_intensity_g_per_kwh"] / 1000.0)
        + row["diesel_kw"] * DIESEL_EF_KG_PER_L / DIESEL_GENSET_KWH_PER_L
    )
    water_l = (
        row["load_kw"] * COOLING_WUE_L_PER_KWH
        + row["grid_kw"] * GRID_UPSTREAM_WATER_L_PER_KWH
        + row["diesel_kw"] * DIESEL_WATER_L_PER_KWH
    )
    cost_inr = row["grid_kw"] * grid_price + row["diesel_kw"] * diesel_price_per_kwh
    return {"co2_kg": co2_kg, "water_l": water_l, "cost_inr": cost_inr}


def compute_baseline(load_kw: np.ndarray, grid_ef: np.ndarray, req: PlanRequest) -> dict:
    """100% grid, no solar/battery/optimisation — the counterfactual the
    dashboard's 'savings' figures are measured against."""
    flat_price = req.grid_price_per_kwh
    co2 = float(np.sum(load_kw * (grid_ef / 1000.0)))
    water = float(np.sum(load_kw * (COOLING_WUE_L_PER_KWH + GRID_UPSTREAM_WATER_L_PER_KWH)))
    cost = float(np.sum(load_kw * flat_price))
    return {"co2_kg": co2, "water_l": water, "cost_inr": cost}


# ============================================================================
# MODEL 5 — ESG NARRATIVE GENERATOR
# Report spec: "A structured prompt to an LLM API (e.g. Claude) that turns
# the computed figures into a plain-language ESG summary paragraph."
# Falls back to a deterministic template if no ANTHROPIC_API_KEY is set, so
# the demo still works offline / without credentials.
# ============================================================================

def generate_esg_narrative(plan: dict) -> ESGNarrativeResponse:
    totals = plan["totals"]
    savings = plan["savings"]
    mix = plan["mix_pct"]

    prompt = (
        "You are drafting one paragraph for a data centre's ESG disclosure summary, "
        "aimed at non-technical stakeholders (board members, compliance officers). "
        "Use the figures below faithfully, plain language, no bullet points, "
        "120-170 words, confident but not promotional.\n\n"
        f"City: {plan['city']}. Planning horizon: {len(plan['hourly'])} hours.\n"
        f"Energy mix: grid {mix['grid_pct']:.1f}%, solar {mix['solar_pct']:.1f}%, "
        f"battery {mix['battery_pct']:.1f}%, diesel {mix['diesel_pct']:.1f}%.\n"
        f"Total CO2 emitted: {totals['co2_kg']:.1f} kg. "
        f"CO2 avoided vs. all-grid baseline: {savings['co2_kg']:.1f} kg "
        f"({savings['co2_pct']:.1f}% reduction).\n"
        f"Total water footprint: {totals['water_l']:.0f} L. "
        f"Water impact change vs. baseline: {savings['water_l']:.0f} L.\n"
        f"Total energy cost: Rs {totals['cost_inr']:.0f}. "
        f"Cost saved vs. baseline: Rs {savings['cost_inr']:.0f} "
        f"({savings['cost_pct']:.1f}% reduction)."
    )

    if ANTHROPIC_API_KEY and ANTHROPIC_SDK_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in resp.content if block.type == "text").strip()
            if text:
                return ESGNarrativeResponse(narrative=text, generated_with=ANTHROPIC_MODEL)
        except Exception:
            pass  # fall through to template

    template = (
        f"Over the {len(plan['hourly'])}-hour planning window for {plan['city']}, the platform's "
        f"optimiser drew {mix['solar_pct']:.0f}% of energy from solar, {mix['battery_pct']:.0f}% from "
        f"battery storage, {mix['grid_pct']:.0f}% from the grid, and {mix['diesel_pct']:.0f}% from diesel "
        f"backup. Compared with a business-as-usual, all-grid baseline, this switching plan avoided an "
        f"estimated {savings['co2_kg']:.1f} kg of CO2 ({savings['co2_pct']:.1f}% lower) and reduced total "
        f"energy cost by roughly Rs {savings['cost_inr']:.0f} ({savings['cost_pct']:.1f}%). Water footprint "
        f"moved by {savings['water_l']:.0f} L against baseline over the same period. These figures are "
        f"generated from forecast and simulated load data for demonstration purposes and should be "
        f"validated against metered site data before being used in a formal ESG disclosure."
    )
    return ESGNarrativeResponse(narrative=template, generated_with="template fallback (no ANTHROPIC_API_KEY set)")


# ============================================================================
# ORCHESTRATION PIPELINE  (Stages 1-3b-4 of the report's system workflow)
# ============================================================================

def run_pipeline(req: PlanRequest) -> dict:
    weather, weather_source = fetch_weather_forecast(req.lat, req.lon, req.hours)

    load_model = LoadForecastModel()
    load_df = load_model.fit_and_predict(req.dc_capacity_kw, req.hours)
    timestamps = load_df["timestamp"].tolist()
    load_kw = load_df["load_kw"].to_numpy()

    solar_model = SolarForecastModel()
    solar_kw = solar_model.predict(weather.iloc[:req.hours], req.solar_capacity_kw)

    grid_ef = grid_carbon_intensity_series(timestamps)
    grid_price = grid_tariff_series(timestamps, req.grid_price_per_kwh)

    alloc_df, engine_name = optimize_sources(load_kw, solar_kw, grid_ef, grid_price, timestamps, req)

    hourly_records = []
    totals = {"co2_kg": 0.0, "water_l": 0.0, "cost_inr": 0.0,
              "grid_kwh": 0.0, "solar_kwh": 0.0, "battery_kwh": 0.0, "diesel_kwh": 0.0}
    diesel_price_per_kwh = req.diesel_price_per_l / DIESEL_GENSET_KWH_PER_L

    for h in range(req.hours):
        row = {
            "load_kw": load_kw[h],
            "grid_kw": alloc_df["grid_kw"].iloc[h],
            "solar_kw": alloc_df["solar_kw"].iloc[h],
            "diesel_kw": alloc_df["diesel_kw"].iloc[h],
            "grid_carbon_intensity_g_per_kwh": grid_ef[h],
        }
        impact = compute_impact_row(pd.Series(row), grid_price[h], diesel_price_per_kwh)
        rec = HourlyRecord(
            hour_index=h,
            timestamp=timestamps[h].isoformat(),
            load_kw=round(float(load_kw[h]), 2),
            solar_available_kw=round(float(solar_kw[h]), 2),
            grid_carbon_intensity_g_per_kwh=round(float(grid_ef[h]), 1),
            grid_price_per_kwh=round(float(grid_price[h]), 2),
            grid_kw=round(float(alloc_df["grid_kw"].iloc[h]), 2),
            solar_kw=round(float(alloc_df["solar_kw"].iloc[h]), 2),
            battery_discharge_kw=round(float(alloc_df["battery_discharge_kw"].iloc[h]), 2),
            battery_charge_kw=round(float(alloc_df["battery_charge_kw"].iloc[h]), 2),
            diesel_kw=round(float(alloc_df["diesel_kw"].iloc[h]), 2),
            battery_soc_kwh=round(float(alloc_df["battery_soc_kwh"].iloc[h]), 2),
            cost_inr=round(float(impact["cost_inr"]), 2),
            co2_kg=round(float(impact["co2_kg"]), 3),
            water_l=round(float(impact["water_l"]), 2),
            grid_outage=bool(alloc_df["grid_outage"].iloc[h]),
        )
        hourly_records.append(rec)
        totals["co2_kg"] += impact["co2_kg"]
        totals["water_l"] += impact["water_l"]
        totals["cost_inr"] += impact["cost_inr"]
        totals["grid_kwh"] += alloc_df["grid_kw"].iloc[h]
        totals["solar_kwh"] += alloc_df["solar_kw"].iloc[h]
        totals["battery_kwh"] += alloc_df["battery_discharge_kw"].iloc[h]
        totals["diesel_kwh"] += alloc_df["diesel_kw"].iloc[h]

    baseline_totals = compute_baseline(load_kw, grid_ef, req)

    def pct_saved(base, actual):
        return 0.0 if base == 0 else 100.0 * (base - actual) / base

    savings = {
        "co2_kg": baseline_totals["co2_kg"] - totals["co2_kg"],
        "co2_pct": pct_saved(baseline_totals["co2_kg"], totals["co2_kg"]),
        "water_l": baseline_totals["water_l"] - totals["water_l"],
        "water_pct": pct_saved(baseline_totals["water_l"], totals["water_l"]),
        "cost_inr": baseline_totals["cost_inr"] - totals["cost_inr"],
        "cost_pct": pct_saved(baseline_totals["cost_inr"], totals["cost_inr"]),
        "diesel_liters": totals["diesel_kwh"] / DIESEL_GENSET_KWH_PER_L,
    }

    total_energy = sum(totals[k] for k in ("grid_kwh", "solar_kwh", "battery_kwh", "diesel_kwh")) or 1.0
    mix_pct = {
        "grid_pct": 100 * totals["grid_kwh"] / total_energy,
        "solar_pct": 100 * totals["solar_kwh"] / total_energy,
        "battery_pct": 100 * totals["battery_kwh"] / total_energy,
        "diesel_pct": 100 * totals["diesel_kwh"] / total_energy,
    }

    plan = {
        "generated_at": datetime.now().isoformat(),
        "city": req.city,
        "lat": req.lat,
        "lon": req.lon,
        "weather_source": weather_source,
        "optimizer_engine": engine_name,
        "hourly": [r.model_dump() for r in hourly_records],
        "totals": totals,
        "baseline_totals": baseline_totals,
        "savings": savings,
        "mix_pct": mix_pct,
    }
    store.save_plan(plan)
    return plan


# ============================================================================
# SCHEDULER  (Stage: hourly forecast-and-recommend cycle, per stack table)
# ============================================================================

scheduler = None
if APSCHEDULER_AVAILABLE:
    scheduler = BackgroundScheduler()

    def _scheduled_run():
        global _last_params
        if _last_params is not None:
            try:
                run_pipeline(_last_params)
            except Exception as e:
                print(f"[scheduler] pipeline run failed: {e}")

    scheduler.add_job(_scheduled_run, "interval", seconds=DEFAULT_INTERVAL_SECONDS, id="hourly_cycle")


# ============================================================================
# FASTAPI APP
# ============================================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    if scheduler is not None and not scheduler.running:
        scheduler.start()
    yield
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="CarbonPilot AI — Energy Source Switch Planner & Carbon-Water Impact Reporter",
    description="Hackathon MVP backend — Green AI Datacenters",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pulp_available": PULP_AVAILABLE,
        "reportlab_available": REPORTLAB_AVAILABLE,
        "anthropic_configured": bool(ANTHROPIC_API_KEY and ANTHROPIC_SDK_AVAILABLE),
        "scheduler_running": bool(scheduler and scheduler.running),
    }


@app.get("/api/config/defaults")
def config_defaults():
    return PlanRequest().model_dump()


@app.post("/api/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest):
    global _last_params
    _last_params = req
    plan = run_pipeline(req)
    return plan


@app.get("/api/plan/latest", response_model=PlanResponse)
def latest_plan():
    plan = store.load_latest()
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan has been generated yet. POST /api/plan first.")
    return plan


@app.get("/api/history")
def history(limit: int = 30):
    return store.load_history(limit)


@app.post("/api/report/esg", response_model=ESGNarrativeResponse)
def esg_report():
    plan = store.load_latest()
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan has been generated yet. POST /api/plan first.")
    return generate_esg_narrative(plan)


@app.get("/api/report/pdf")
def esg_report_pdf():
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=500, detail="reportlab is not installed on the server.")
    plan = store.load_latest()
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan has been generated yet. POST /api/plan first.")
    narrative = generate_esg_narrative(plan)

    tmp_path = Path(tempfile.gettempdir()) / f"esg_report_{int(datetime.now().timestamp())}.pdf"
    _build_pdf(plan, narrative, tmp_path)
    return FileResponse(tmp_path, filename="esg_impact_report.pdf", media_type="application/pdf")


def _build_pdf(plan: dict, narrative: ESGNarrativeResponse, out_path: Path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=colors.HexColor("#1B3A2B"))
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], leading=15)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    elements = [
        Paragraph("ESG Impact Report", title_style),
        Paragraph("CarbonPilot AI — Energy Source Switch Planner &amp; Carbon-Water Impact Reporter", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph(f"Location: {plan['city']}  |  Generated: {plan['generated_at']}  |  "
                  f"Horizon: {len(plan['hourly'])}h  |  Engine: {plan['optimizer_engine']}", styles["Italic"]),
        Spacer(1, 6 * mm),
        Paragraph("Executive Summary", styles["Heading2"]),
        Paragraph(narrative.narrative, body_style),
        Spacer(1, 6 * mm),
        Paragraph("Key Figures", styles["Heading2"]),
    ]

    totals, baseline, savings, mix = plan["totals"], plan["baseline_totals"], plan["savings"], plan["mix_pct"]
    table_data = [
        ["Metric", "Optimised Plan", "All-Grid Baseline", "Savings"],
        ["CO2 emissions (kg)", f"{totals['co2_kg']:.1f}", f"{baseline['co2_kg']:.1f}", f"{savings['co2_kg']:.1f} ({savings['co2_pct']:.1f}%)"],
        ["Water footprint (L)", f"{totals['water_l']:.0f}", f"{baseline['water_l']:.0f}", f"{savings['water_l']:.0f} ({savings['water_pct']:.1f}%)"],
        ["Energy cost (Rs)", f"{totals['cost_inr']:.0f}", f"{baseline['cost_inr']:.0f}", f"{savings['cost_inr']:.0f} ({savings['cost_pct']:.1f}%)"],
        ["Diesel used (L)", f"{savings['diesel_liters']:.1f}", "-", "-"],
    ]
    tbl = Table(table_data, colWidths=[45 * mm, 38 * mm, 38 * mm, 45 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B3A2B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F3")]),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph("Energy Mix Over Planning Horizon", styles["Heading2"]))
    mix_data = [
        ["Grid", "Solar", "Battery", "Diesel"],
        [f"{mix['grid_pct']:.1f}%", f"{mix['solar_pct']:.1f}%", f"{mix['battery_pct']:.1f}%", f"{mix['diesel_pct']:.1f}%"],
    ]
    mix_tbl = Table(mix_data, colWidths=[41.5 * mm] * 4)
    mix_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0EA")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(mix_tbl)
    elements.append(Spacer(1, 8 * mm))
    elements.append(Paragraph(
        "Figures are generated from a forecast + simulation pipeline for demonstration purposes "
        "(hackathon MVP scope) and should be validated against metered site data before use in a "
        "formal ESG disclosure.", styles["Italic"]))

    doc.build(elements)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
