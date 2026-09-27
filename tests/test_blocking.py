"""Unit tests for blocking and indexing."""

from src.blocking.block_keys import (
    get_address_number_keys,
    get_clean_name_key,
    get_exact_name_key,
    get_prefix_key,
    get_token_keys,
)


def test_get_exact_name_key():
    key = get_exact_name_key("ABC Technologies Pvt. Ltd.", "India")
    assert key == "india::abc technology private limited"


def test_get_clean_name_key():
    key = get_clean_name_key("Amazon Web Services, Inc.", "USA")
    assert key == "us::amazon web services"


def test_get_prefix_key():
    key = get_prefix_key("Reliance Retail Ltd", "India", length=4)
    assert key == "india::reli"


def test_get_token_keys():
    tokens = get_token_keys("Starbucks Coffee Company", "US", min_len=4)
    assert "us::starbucks" in tokens
    assert "us::coffee" in tokens


def test_get_address_number_keys():
    num_keys = get_address_number_keys("123 Main St, Suite 400, Seattle 98101", "USA")
    assert "us::123" in num_keys
    assert "us::400" in num_keys
    assert "us::98101" in num_keys
