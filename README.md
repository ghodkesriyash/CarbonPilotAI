# CarbonPilot AI — Energy Source Switch Planner & Carbon-Water Impact Reporter

A working prototype of the hackathon MVP described in the technical report and TIH
proposal: an AI planner that recommends the lowest-cost/lowest-carbon energy source
(grid / solar / battery / diesel) for a data centre hour-by-hour, then turns that
plan into a carbon & water impact report.

```
dc-planner/
├── backend/
│   ├── app.py            <- single-file backend: all 5 models + API
│   ├── requirements.txt
│   └── data/             <- created automatically (local JSON "database")
├── frontend/
│   └── index.html        <- single-file UI (HTML + CSS + JS, no build step)
└── README.md
```

## 1. Run the backend

Requires Python 3.10+.

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API starts at **http://localhost:8000**. Interactive API docs (Swagger UI) are
available at http://localhost:8000/docs — useful for poking at the five models
directly.

### Optional: real ESG narratives via Claude

Without any setup, the ESG narrative endpoint uses a deterministic template so the
demo works fully offline. To have Claude write the narrative instead, set an API
key before starting the server:

```bash
cp .env.example .env
# edit .env and paste your key
export $(cat .env | xargs)   # or use python-dotenv / your shell's env loading
python app.py
```

`ANTHROPIC_MODEL` defaults to `claude-sonnet-5`; change it in `.env` if you want a
different model.

### Optional: live weather

The solar forecasting model calls [Open-Meteo](https://open-meteo.com) (free, no
API key) for real cloud cover / irradiance. If there's no network access, it falls
back automatically to a synthetic clear-sky model — the pipeline still runs end to
end either way. Check `weather_source` in the `/api/plan` response to see which one
was used.

## 2. Open the frontend

No build step — it's a single static HTML file.

```bash
cd frontend
python -m http.server 5500
# then open http://localhost:5500 in your browser
```

(or just double-click `index.html` to open it directly in a browser — CORS is
already enabled on the backend for this).

If your backend isn't running on `localhost:8000`, change the **Backend API base
URL** field under "Advanced" in the dashboard's config panel.

## 3. Using it

1. Pick a site, planning horizon, and drag the cost ↔ carbon priority slider.
2. Optionally simulate a grid outage — this is the only condition under which the
   optimiser will call on diesel backup.
3. Click **Run the planner**. This hits the real backend: it forecasts load and
   solar for the horizon, solves a linear program over the whole window (coupled
   through battery state of charge), and returns the hourly plan plus its carbon,
   water, and cost footprint versus an all-grid baseline.
4. Scroll to **ESG impact report** and click **Generate ESG narrative**, then
   **Download PDF report** for a shareable one-pager.

## What's simulated vs. real

Per the report's own scoping (Section 7, Data Sources):

| Data | Status |
|---|---|
| Weather / solar irradiance | Real (Open-Meteo), synthetic fallback if offline |
| Data-centre load | Synthetic diurnal profile (real DC telemetry not available for a hackathon team — clearly labelled per the report) |
| Grid carbon intensity | Simplified diurnal model around the published CEA baseline average (swap in Electricity Maps / WattTime / Grid-India for production) |
| Emission & water factors | Published IPCC / CEA figures and industry WUE benchmarks |
| Source optimisation | A genuine linear program (PuLP + CBC), not a mock |

## Models implemented (see `app.py` section headers)

1. **Load forecasting** — Fourier-feature linear regression (daily + weekly
   seasonality), a lightweight stand-in for Prophet/SARIMA per the report's
   stretch-goal note.
2. **Solar forecasting** — regression on weather features (cloud cover,
   irradiance).
3. **Source optimisation** — linear program across the full horizon, coupling
   every hour through battery SOC; falls back to a rule-based greedy allocator if
   PuLP is unavailable.
4. **Carbon & water calculator** — deterministic, using published emission/water
   factors.
5. **ESG narrative generator** — Claude API call with a template fallback.

## Post-hackathon roadmap (not in this MVP, per the report's Section 9)

Cooling optimisation, full IoT/BMS integration, multi-site fleet dashboard, carbon
credit calculation, and a reinforcement-learning switching policy are deliberately
out of scope here.
