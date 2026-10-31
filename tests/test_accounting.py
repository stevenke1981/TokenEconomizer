"""Ledger summary: how folded sessions were priced, so a fallback-priced total is visible."""
import pytest

from tokeneconomizer.plugin import accounting


def _session(model, usage_in, **extra):
    eff, by = accounting.effective({"input_tokens": usage_in, "output_tokens": 0,
                                    "model": model} if usage_in is not None else None)
    return dict({"kind": "session", "model": model, "effective_tokens": eff,
                 "priced_by": by}, **extra)


def test_a_model_missing_from_the_price_table_is_counted_as_fallback():
    recs = [_session("claude-sonnet-4-5", 1000), _session("not-a-real-model-x", 3000),
            _session(None, None)]
    cov = accounting.summarize(recs)["pricing"]
    assert cov["by_source"]["model"] == {"sessions": 1, "effective_tokens": 1000}
    assert cov["by_source"]["fallback"] == {"sessions": 1, "effective_tokens": 3000}
    assert cov["by_source"]["no_usage"]["sessions"] == 1
    assert cov["fallback_models"] == ["not-a-real-model-x"]
    assert cov["fallback_frac"] == pytest.approx(0.75)
    assert cov["by_model"] == {"claude-sonnet-4-5": 1000, "not-a-real-model-x": 3000}


def test_records_from_before_priced_by_count_as_unknown_not_as_priced():
    cov = accounting.summarize([{"kind": "session", "model": "m", "effective_tokens": 50}])
    assert cov["pricing"]["by_source"] == {"unknown": {"sessions": 1, "effective_tokens": 50}}
    assert cov["pricing"]["fallback_frac"] == 0.0


def test_an_empty_ledger_has_no_coverage_to_report():
    cov = accounting.summarize([])["pricing"]
    assert cov == {"by_source": {}, "by_model": {}, "fallback_models": [],
                   "fallback_frac": 0.0}
