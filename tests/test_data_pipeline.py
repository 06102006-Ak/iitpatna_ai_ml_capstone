from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
from scrape_books import clean_books, make_offline_fixture, parse_price, parse_rating, parse_stock  # noqa: E402


def test_field_parsers():
    assert parse_price("£12.34") == 12.34
    assert parse_rating("Five") == 5
    assert parse_stock("In stock (19 available)") is True
    assert parse_stock("Out of stock") is False


def test_offline_fixture_meets_minimum():
    df = make_offline_fixture(100)
    cleaned, _ = clean_books(df)
    assert len(cleaned) >= 60
    assert cleaned["category"].nunique() >= 3
    assert cleaned["rating"].between(1, 5).all()
    assert cleaned["in_stock"].dtype == bool
    assert (cleaned["price_inr"] == (cleaned["price_gbp"] * 105.5).round(2)).all()
