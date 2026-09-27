"""Blocking key generator functions."""
# pyrefly: ignore [missing-import]
from src.preprocessing.address_normalizer import extract_numbers

# pyrefly: ignore [missing-import]
from src.preprocessing.country_normalizer import normalize_country

# pyrefly: ignore [missing-import]
from src.preprocessing.name_normalizer import extract_name_tokens, normalize_business_name, remove_legal_suffixes


def get_exact_name_key(name: str, country: str) -> str:
    """Exact normalized business name blocking key."""
    norm_name = normalize_business_name(name)
    if not norm_name:
        return ""
    norm_country = normalize_country(country)
    return f"{norm_country}::{norm_name}"


def get_clean_name_key(name: str, country: str) -> str:
    """Business name stripped of legal suffixes."""
    clean_name = remove_legal_suffixes(name)
    if not clean_name:
        return ""
    norm_country = normalize_country(country)
    return f"{norm_country}::{clean_name}"


def get_prefix_key(name: str, country: str, length: int = 4) -> str:
    """First N characters of cleaned business name."""
    clean_name = remove_legal_suffixes(name).replace(" ", "")
    if len(clean_name) < length:
        return ""
    norm_country = normalize_country(country)
    return f"{norm_country}::{clean_name[:length]}"


def get_token_keys(name: str, country: str, min_len: int = 3) -> list[str]:
    """Significant tokens from business name."""
    tokens = extract_name_tokens(name)
    norm_country = normalize_country(country)
    return [f"{norm_country}::{t}" for t in tokens if len(t) >= min_len]


def get_address_number_keys(address: str, country: str) -> list[str]:
    """Extracted numeric sequences (postal codes/house numbers) combined with country."""
    numbers = extract_numbers(address)
    norm_country = normalize_country(country)
    return [f"{norm_country}::{num}" for num in numbers if len(num) >= 3]


if __name__ == "__main__":
    test_cases = [
        {
            "name": "ABC Technologies Pvt. Ltd.",
            "country": "India",
            "address": "Plot No. 42, Sector 18, Gurugram, Haryana 122015",
        },
        {
            "name": "Amazon Web Services, Inc.",
            "country": "USA",
            "address": "410 Terry Ave N, Seattle, WA 98109",
        },
        {
            "name": "Café de la Gare SARL",
            "country": "France",
            "address": "15 Rue de la Paix, 75002 Paris",
        },
    ]

    print("=" * 70)
    print("BLOCKING KEYS DEMO RUN")
    print("=" * 70)

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Entity #{i} ---")
        print(f"Original Name:    '{tc['name']}'")
        print(f"Original Country: '{tc['country']}'")
        print(f"Original Address: '{tc['address']}'")
        print(f"-> Exact Name Key:     {get_exact_name_key(tc['name'], tc['country'])}")
        print(f"-> Clean Name Key:     {get_clean_name_key(tc['name'], tc['country'])}")
        print(f"-> Prefix (4) Key:     {get_prefix_key(tc['name'], tc['country'], length=4)}")
        print(f"-> Token Keys:         {get_token_keys(tc['name'], tc['country'])}")
        print(f"-> Address Number Keys: {get_address_number_keys(tc['address'], tc['country'])}")

    print("\n" + "=" * 70)
    print("SUCCESS: All blocking key generators executed successfully!")
    print("=" * 70)
