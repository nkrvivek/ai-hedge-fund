"""Pydantic models for event study results.

These models capture every layer of the event study pipeline:

    MarketModelFit   — one OLS regression (α, β from the estimation window)
    EventCAR         — one event's CARs (single ticker × single earnings filing)
    WindowStats      — aggregate stats for one CAR window across many events
    AggregateResult  — all WindowStats for one source_type group (e.g. all 8-K events)
    EventStudyResult — top-level container returned by compute_car()
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketModelFit(BaseModel):
    """OLS regression result: R_stock = alpha + beta * R_market.

    Fit over the estimation window [-250, -11] trading days before an event.
    alpha = stock's baseline daily excess return (intercept).
    beta  = stock's sensitivity to market moves (slope).
    """

    alpha: float           # intercept — expected daily return when market return is 0
    beta: float            # slope — how much the stock amplifies market moves
    r_squared: float       # fraction of stock variance explained by the market
    n_obs: int             # number of trading days in the estimation window


class EventCAR(BaseModel):
    """CAR result for a single event (one ticker, one earnings filing).

    Each earnings filing (8-K, 10-Q, 10-K) produces one EventCAR.
    The same report_period may yield multiple EventCARs if both an 8-K
    and a 10-Q exist for that quarter.
    """

    ticker: str
    event_date: str                           # filing_date used as event anchor (YYYY-MM-DD)
    source_type: str                          # "8-K", "10-Q", "10-K", "20-F"
    report_period: str                        # fiscal quarter end date
    eps_surprise: str | None = None           # "BEAT" / "MISS" / "MEET" from quarterly data
    market_model: MarketModelFit              # the α, β used for this event
    daily_ar: list[float] = Field(default_factory=list)  # AR for each day [0, +20]
    car_0_1: float | None = None              # cumulative AR over [0, +1] (2 days)
    car_0_5: float | None = None              # cumulative AR over [0, +5] (6 days)
    car_0_20: float | None = None             # cumulative AR over [0, +20] (21 days)


class BootstrapCI(BaseModel):
    """Bootstrap confidence interval for mean CAR.

    Built by resampling the observed CARs with replacement N times,
    computing the mean of each resample, and taking percentiles.
    """

    lower: float                              # lower bound of CI
    upper: float                              # upper bound of CI
    confidence: float = 0.95                  # confidence level (default 95%)
    n_bootstrap: int = 10_000                 # number of bootstrap resamples


class WindowStats(BaseModel):
    """Aggregate statistics for one CAR window across many events.

    Example: mean CAR[0,+1] across all 8-K events, with t-test and bootstrap CI.
    If mean_car is significantly different from 0, the event type moves prices.
    """

    window: str                               # human label: "[0,+1]", "[0,+5]", "[0,+20]"
    n_events: int                             # number of events with non-null CAR for this window
    mean_car: float                           # average CAR across events
    std_car: float                            # standard deviation of CARs (sample, ddof=1)
    t_stat: float                             # one-sample t-stat vs H0: mean = 0
    p_value: float                            # two-sided p-value from t-test
    ci: BootstrapCI                           # bootstrap 95% CI for the mean


class AggregateResult(BaseModel):
    """Cross-sectional results for one source_type (e.g. all 8-K events).

    Segmenting by source_type is a built-in robustness check:
    8-K-anchored events should show stronger/faster CARs than 10-Q/K-derived,
    because 8-K filing_date is closer to the actual announcement.
    """

    source_type: str                          # "8-K", "10-Q", "10-K", "20-F"
    n_events: int                             # total events in this group
    windows: list[WindowStats] = Field(default_factory=list)


class EstimationWindow(BaseModel):
    """The estimation window the run actually used.

    The configured window is 250 trading days deep. A feed that holds one
    year cannot support it, so the run narrows the window to what the data
    allows. That changes alpha, beta and every CAR computed from them, which
    makes it a stated choice rather than something to discover later from a
    surprising number.
    """

    start: int             # trading days before the event, negative
    end: int               # last day of the window, also negative
    n_days: int            # end - start + 1
    min_days: int          # fewest observations an event may be fitted on
    narrowed: bool         # True when the feed forced a shallower window
    reason: str            # what the run did and why


class EventStudyResult(BaseModel):
    """Top-level result returned by compute_car().

    Contains per-event detail (events), cross-sectional aggregates
    (segmented by source_type), the estimation window that was used, and
    why each skipped ticker was skipped.

    skip_reasons exists because skipped_tickers carried two meanings that
    printed the same. A company with no earnings filings and a company the
    feed could not cover are different findings, and reading the first as
    the second is how a data fault gets filed as a market observation.
    """

    events: list[EventCAR] = Field(default_factory=list)
    aggregates: list[AggregateResult] = Field(default_factory=list)
    skipped_tickers: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)
    estimation_window: EstimationWindow | None = None
