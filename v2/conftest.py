"""Load .env for v2 tests so FINANCIAL_DATASETS_API_KEY is available."""

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def _off_the_live_fmp_counter(tmp_path, monkeypatch):
    """Keep the suite out of ~/.fmp-spend.

    v2/data/fmp_spend.py writes every FMP response's byte count to a shared
    on-disk counter that trade-refresh's src/fmp_quota.py reads across five
    consumers. A test that fakes a 429 would otherwise mark the live day as
    rate-limited, and the rail would report FMP turning us away because pytest
    ran.
    """
    monkeypatch.setenv("FMP_SPEND_DIR", str(tmp_path / "fmp-spend"))
