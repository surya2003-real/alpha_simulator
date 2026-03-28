"""
app.py  –  AlphaSim FastAPI backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import traceback

from engine import BacktestEngine, BacktestConfig, FactorConfig, FACTOR_COLS, HIGHER_IS_BETTER
from data_loader import SECTORS, MARKET_CAPS

app = FastAPI(title="AlphaSim API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve static frontend ─────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ─── Pydantic models ───────────────────────────

class FactorIn(BaseModel):
    name: str
    weight: float = Field(gt=0, le=1)
    higher_is_better: bool = True
    winsorize: bool = True
    zscore: bool = True


class BacktestRequest(BaseModel):
    start_date:     str = "2004-01-01"
    end_date:       str = "2018-12-31"
    custom_alpha:   Optional[str] = None  # <--- MUST HAVE THIS
    factors:        List[FactorIn]
    sectors:        Optional[List[str]] = None
    market_caps:    Optional[List[str]] = None
    min_adv:        float = 0.0
    max_positions:  int = 50
    max_weight:     float = 0.05
    weighting:      str = "equal"
    long_short:     bool = False
    short_fraction: float = 0.2
    rebalance_freq: str = "M"
    slippage_bps:   float = 10.0
    commission_bps: float = 5.0
    max_drawdown_stop: float = 0.20
    benchmark:      str = "SP500"


# ─── Endpoints ─────────────────────────────────

@app.get("/api/meta")
def get_meta():
    """Return available factors, sectors, caps."""
    return {
        "factors": [
            {"col": k, "label": v, "higher_is_better": HIGHER_IS_BETTER[k]}
            for k, v in FACTOR_COLS.items()
        ],
        "sectors":     SECTORS,
        "market_caps": MARKET_CAPS,
    }


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Run a full backtest and return results."""
    try:
        t0 = time.time()

        # <--- UPDATED LOGIC: Allow custom alpha OR factors
        if not req.factors and not req.custom_alpha:
            raise HTTPException(400, "At least one factor or a custom alpha is required.")

        # Normalise weights safely (prevent divide by zero if no factors are passed)
        total_w = sum(f.weight for f in req.factors) if req.factors else 1.0
        factors = [
            FactorConfig(
                name=f.name,
                weight=f.weight / total_w,
                higher_is_better=f.higher_is_better,
                winsorize=f.winsorize,
                zscore=f.zscore,
            )
            for f in req.factors
        ]

        cfg = BacktestConfig(
            start_date=req.start_date,
            end_date=req.end_date,
            custom_alpha=req.custom_alpha, # <--- MUST PASS TO ENGINE
            factors=factors,
            sectors=req.sectors or None,
            market_caps=req.market_caps or None,
            min_adv=req.min_adv,
            max_positions=req.max_positions,
            max_weight=req.max_weight,
            weighting=req.weighting,
            long_short=req.long_short,
            short_fraction=req.short_fraction,
            rebalance_freq=req.rebalance_freq,
            slippage_bps=req.slippage_bps,
            commission_bps=req.commission_bps,
            max_drawdown_stop=req.max_drawdown_stop,
            benchmark=req.benchmark,
        )

        engine = BacktestEngine(cfg)
        results = engine.run()
        results["elapsed_s"] = round(time.time() - t0, 2)
        return results

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Backtest error: {str(e)}")


@app.get("/api/health")
def health():
    return {"status": "ok"}