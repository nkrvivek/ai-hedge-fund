from bridge.run_daily import composite, rebalance_orders, target_weights


def test_composite_ignores_abstains():
    assert composite([0.6, 0.0]) == 0.6
    assert composite([0.0, 0.0]) == 0.0
    assert composite([0.5, -0.5]) == 0.0


def test_weights_capped_per_name_and_gross():
    w = target_weights({"A": 1.0, "B": -1.0, "C": 1.0})
    assert all(abs(x) <= 0.10 for x in w.values())
    assert sum(abs(x) for x in w.values()) <= 1.0 + 1e-9


def test_rebalance_diffs_and_dust_filter():
    orders = rebalance_orders({"A": 0.10}, {"A": 9000.0, "B": 5000.0}, 100_000)
    by = {o["symbol"]: o for o in orders}
    assert by["A"]["side"] == "buy" and by["A"]["delta_usd"] == 1000.0
    assert by["B"]["side"] == "sell" and by["B"]["delta_usd"] == -5000.0
    assert not rebalance_orders({"A": 0.05}, {"A": 4900.0}, 100_000)  # dust


def test_paper_endpoint_guard(monkeypatch):
    import pytest
    from bridge.alpaca import AlpacaPaper, AlpacaPaperError
    monkeypatch.setenv("ALPACA_TRADING_ENDPOINT", "https://api.alpaca.markets/v2")
    with pytest.raises(AlpacaPaperError):
        AlpacaPaper()
