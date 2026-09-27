"""Unit tests for text, name, address, and country normalization."""
from src.preprocessing.address_normalizer import extract_numbers, normalize_address
from src.preprocessing.country_normalizer import is_country_match, normalize_country
from src.preprocessing.name_normalizer import extract_name_tokens, normalize_business_name, remove_legal_suffixes
from src.preprocessing.text_normalizer import normalize_text_base


def test_text_normalization_unicode():
    raw = "Café  de   la  Gare\xa0"
    norm = normalize_text_base(raw)
    assert norm == "cafe de la gare"


def test_name_normalization_legal_suffixes():
    name = "Reliance Retail Pvt. Ltd."
    norm = normalize_business_name(name)
    assert norm == "reliance retail private limited"
    clean = remove_legal_suffixes(name)
    assert clean == "reliance retail"


def test_name_tokens():
    tokens = extract_name_tokens("Amazon Web Services LLC")
    assert "amazon" in tokens
    assert "web" in tokens
    assert "services" in tokens


def test_address_normalization():
    addr = "123 Main St., Suite 400, Seattle Rd."
    norm = normalize_address(addr)
    assert "street" in norm
    assert "suite" in norm
    assert "road" in norm


def test_extract_numbers():
    addr = "Plot No. 42, Sector 18, Gurugram 122015"
    nums = extract_numbers(addr)
    assert "42" in nums
    assert "18" in nums
    assert "122015" in nums


def test_country_normalization_open_set():
    assert normalize_country("US") == "us"
    assert normalize_country("United States") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("France") == "france"
    assert normalize_country("Germany") == "germany"  # open set, never crashes
    assert is_country_match("USA", "United States") is True
    assert is_country_match("India", "France") is False
