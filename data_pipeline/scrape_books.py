"""
Book scraping and cleaning pipeline for the capstone project.

Source:
    https://books.toscrape.com/

The scraper:
- Uses requests + BeautifulSoup
- Scrapes multiple catalogue categories
- Captures title, price, rating, availability, and category
- Cleans numeric/boolean fields
- Converts GBP to INR using the required fixed rate
- Validates the final dataset
- Supports an offline fixture for reproducibility/testing

Usage:
    python data_pipeline/scrape_books.py
    python data_pipeline/scrape_books.py --pages 5
    python data_pipeline/scrape_books.py --offline-fixture
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

# We intentionally scrape several categories so the final dataset
# satisfies the assignment requirement of >= 3 categories.
TARGET_CATEGORIES = [
    "Travel",
    "Mystery",
    "Historical Fiction",
    "Poetry",
    "Science Fiction",
]


# ---------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------

def build_session() -> requests.Session:
    """
    Create a requests session with retries for temporary server errors.
    """

    session = requests.Session()

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            )
        }
    )

    return session


# ---------------------------------------------------------------------
# Request helper
# ---------------------------------------------------------------------

def fetch_soup(
    session: requests.Session,
    url: str,
    timeout: int = 30,
) -> BeautifulSoup:
    """
    Download a webpage and return its BeautifulSoup representation.
    """

    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# ---------------------------------------------------------------------
# Category discovery
# ---------------------------------------------------------------------

def discover_category_urls(
    session: requests.Session,
) -> Dict[str, str]:
    """
    Discover category URLs from the Books to Scrape homepage.

    Returns:
        Dictionary mapping category name -> absolute URL.
    """

    soup = fetch_soup(session, BASE_URL)

    category_urls: Dict[str, str] = {}

    nav = soup.select("ul.nav-list ul li a")

    for link in nav:
        category_name = link.get_text(" ", strip=True)

        if category_name in TARGET_CATEGORIES:
            href = link.get("href")

            if href:
                category_urls[category_name] = urljoin(
                    BASE_URL,
                    href,
                )

    return category_urls


# ---------------------------------------------------------------------
# Rating parser
# ---------------------------------------------------------------------

def parse_rating(article) -> Optional[int]:
    """
    Extract the numerical rating from the star-rating CSS class.
    """

    rating_element = article.select_one("p.star-rating")

    if rating_element is None:
        return None

    classes = rating_element.get("class", [])

    for class_name in classes:
        if class_name in RATING_MAP:
            return RATING_MAP[class_name]

    return None


# ---------------------------------------------------------------------
# Availability parser
# ---------------------------------------------------------------------

def parse_availability(article) -> Optional[str]:
    """
    Extract the availability text.
    """

    availability_element = article.select_one(
        "p.instock.availability"
    )

    if availability_element is None:
        return None

    return availability_element.get_text(" ", strip=True)


# ---------------------------------------------------------------------
# Price parser
# ---------------------------------------------------------------------

def parse_price(text: str) -> Optional[float]:
    """
    Extract a GBP price from text.

    Example:
        '£51.77' -> 51.77
    """

    if not text:
        return None

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Listing page parser
# ---------------------------------------------------------------------

def parse_listing_page(
    soup: BeautifulSoup,
    page_url: str,
    category: str,
) -> List[dict]:
    """
    Parse one category listing page.

    Each product is represented by a dictionary.
    """

    books: List[dict] = []

    articles = soup.select("article.product_pod")

    for article in articles:

        title_element = article.select_one("h3 a")

        price_element = article.select_one(
            "p.price_color"
        )

        if title_element is None:
            continue

        title = title_element.get("title")

        if not title:
            title = title_element.get_text(
                " ",
                strip=True,
            )

        detail_href = title_element.get("href")

        detail_url = (
            urljoin(page_url, detail_href)
            if detail_href
            else None
        )

        price_text = (
            price_element.get_text(
                " ",
                strip=True,
            )
            if price_element
            else None
        )

        price_gbp = parse_price(price_text)

        rating = parse_rating(article)

        availability = parse_availability(article)

        in_stock = None

        if availability:
            in_stock = (
                "in stock"
                in availability.lower()
            )

        books.append(
            {
                "title": title,
                "price_gbp": price_gbp,
                "rating": rating,
                "availability": availability,
                "in_stock": in_stock,
                "category": category,
                "detail_url": detail_url,
                "listing_page": page_url,
            }
        )

    return books


# ---------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------

def next_page_url(
    soup: BeautifulSoup,
    current_url: str,
) -> Optional[str]:
    """
    Return the next category page URL if available.
    """

    next_link = soup.select_one("li.next a")

    if next_link is None:
        return None

    href = next_link.get("href")

    if not href:
        return None

    return urljoin(current_url, href)


# ---------------------------------------------------------------------
# Category scraper
# ---------------------------------------------------------------------

def scrape_categories(
    session: requests.Session,
    category_urls: Dict[str, str],
    delay: float = 0.75,
) -> List[dict]:
    """
    Scrape all selected categories.

    Pagination continues until a category has no next page.
    """

    all_books: List[dict] = []

    for category, start_url in category_urls.items():

        print(
            f"\nScraping category: {category}"
        )

        current_url: Optional[str] = start_url

        page_number = 1

        while current_url:

            print(
                f"  Page {page_number}: "
                f"{current_url}"
            )

            try:
                soup = fetch_soup(
                    session,
                    current_url,
                )

                page_books = parse_listing_page(
                    soup=soup,
                    page_url=current_url,
                    category=category,
                )

                print(
                    f"    Books found: "
                    f"{len(page_books)}"
                )

                all_books.extend(page_books)

                current_url = next_page_url(
                    soup,
                    current_url,
                )

                page_number += 1

                if current_url:
                    time.sleep(delay)

            except requests.RequestException as exc:
                print(
                    f"    Request failed: {exc}"
                )

                break

    return all_books


# ---------------------------------------------------------------------
# Offline fixture
# ---------------------------------------------------------------------

def build_offline_fixture() -> List[dict]:
    """
    Generate deterministic offline data.

    This is useful for testing the pipeline when internet access is
    unavailable.

    It creates 100 records across 5 categories.
    """

    categories = [
        "Travel",
        "Mystery",
        "Historical Fiction",
        "Poetry",
        "Science Fiction",
    ]

    books: List[dict] = []

    for index in range(100):

        category = categories[index % len(categories)]

        rating = (index % 5) + 1

        price = round(
            5.50 + (index % 30) * 1.25,
            2,
        )

        books.append(
            {
                "title": (
                    f"Offline Sample Book "
                    f"{index + 1}"
                ),
                "price_gbp": price,
                "rating": rating,
                "availability": "In stock",
                "in_stock": True,
                "category": category,
                "detail_url": (
                    f"https://example.com/books/"
                    f"{index + 1}"
                ),
                "listing_page": (
                    f"https://example.com/"
                    f"{category.lower().replace(' ', '-')}"
                ),
            }
        )

    return books


# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------

def clean_books(
    books: List[dict],
) -> tuple[pd.DataFrame, dict]:
    """
    Clean scraped records.

    Cleaning requirements:
    - price_gbp -> float
    - rating -> int 1-5
    - in_stock -> bool
    - price_inr -> fixed conversion
    - invalid rows are dropped where essential fields cannot be fixed
    """

    df = pd.DataFrame(books)

    if df.empty:
        raise ValueError(
            "No books were scraped."
        )

    # -------------------------------------------------------------
    # Normalize price
    # -------------------------------------------------------------

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Normalize rating
    # -------------------------------------------------------------

    def normalize_rating(value):
        if pd.isna(value):
            return None

        if isinstance(value, str):
            value = value.strip()

            if value in RATING_MAP:
                return RATING_MAP[value]

        try:
            value = int(value)

            if 1 <= value <= 5:
                return value

        except (TypeError, ValueError):
            pass

        return None

    df["rating"] = df["rating"].apply(
        normalize_rating
    )

    # -------------------------------------------------------------
    # Normalize stock flag
    # -------------------------------------------------------------

    def normalize_stock(value):
        if isinstance(value, bool):
            return value

        if pd.isna(value):
            return None

        text = str(value).strip().lower()

        if "in stock" in text:
            return True

        if "out of stock" in text:
            return False

        if text in {"true", "1", "yes"}:
            return True

        if text in {"false", "0", "no"}:
            return False

        return None

    df["in_stock"] = df["in_stock"].apply(
        normalize_stock
    )

    # -------------------------------------------------------------
    # Numeric imputation
    # -------------------------------------------------------------

    numeric_imputations = {}

    for column in [
        "price_gbp",
        "rating",
    ]:

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count > 0:

            median_value = df[column].median()

            if pd.notna(median_value):

                if column == "rating":
                    median_value = int(
                        round(median_value)
                    )

                df[column] = df[column].fillna(
                    median_value
                )

                numeric_imputations[column] = (
                    float(median_value)
                )

    # -------------------------------------------------------------
    # Drop rows where essential fields remain invalid
    # -------------------------------------------------------------

    before_rows = len(df)

    df = df.dropna(
        subset=[
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category",
        ]
    ).copy()

    dropped_rows = before_rows - len(df)

    # -------------------------------------------------------------
    # Ensure correct data types
    # -------------------------------------------------------------

    df["price_gbp"] = df["price_gbp"].astype(float)

    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    df["in_stock"] = df["in_stock"].astype(bool)

    df["title"] = (
        df["title"]
        .astype(str)
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------
    # Fixed GBP -> INR conversion
    # -------------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"]
        * FIXED_GBP_TO_INR
    ).round(2)

    # -------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------

    columns = [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "availability",
        "in_stock",
        "category",
        "detail_url",
        "listing_page",
    ]

    df = df[
        [
            column
            for column in columns
            if column in df.columns
        ]
    ]

    cleaning_report = {
        "input_rows": int(before_rows),
        "output_rows": int(len(df)),
        "dropped_rows": int(dropped_rows),
        "numeric_imputations": (
            numeric_imputations
        ),
    }

    return df, cleaning_report


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_final_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate the final cleaned dataset against the assignment
    requirements.
    """

    if len(df) < 60:
        raise ValueError(
            "Final dataset must contain at least "
            "60 rows."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "Final dataset must contain at least "
            "3 categories."
        )

    if df["price_gbp"].isna().any():
        raise ValueError(
            "price_gbp contains missing values."
        )

    if df["rating"].isna().any():
        raise ValueError(
            "rating contains missing values."
        )

    if not df["rating"].between(
        1,
        5,
    ).all():
        raise ValueError(
            "rating must contain only integers "
            "from 1 to 5."
        )

    if df["in_stock"].isna().any():
        raise ValueError(
            "in_stock contains missing values."
        )

    if df["in_stock"].dtype != bool:
        raise ValueError(
            "in_stock must be boolean."
        )

    # Verify fixed conversion exactly.
    expected_inr = (
        df["price_gbp"]
        * FIXED_GBP_TO_INR
    ).round(2)

    if not (
        df["price_inr"].round(2)
        == expected_inr
    ).all():
        raise ValueError(
            "price_inr does not use the required "
            "fixed conversion rate."
        )

    # Unknown category is not acceptable for the
    # final submitted dataset.
    unknown_mask = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "unknown"
    )

    if unknown_mask.any():
        raise ValueError(
            "Final dataset contains Unknown "
            "categories."
        )


# ---------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------

def save_report(
    report_path: Path,
    scrape_stats: dict,
    cleaning_report: dict,
    df: pd.DataFrame,
) -> None:
    """
    Save pipeline statistics as JSON.
    """

    report = {
        "scrape_stats": scrape_stats,
        "cleaning": cleaning_report,
        "fixed_gbp_to_inr": FIXED_GBP_TO_INR,
        "final_rows": int(len(df)),
        "categories": int(
            df["category"].nunique()
        ),
        "category_counts": {
            str(k): int(v)
            for k, v in (
                df["category"]
                .value_counts()
                .to_dict()
                .items()
            )
        },
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Scrape and clean books from "
            "Books to Scrape."
        )
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # run_pipeline.py already sends --pages 5.
    # We retain this argument for compatibility.
    # The category-based scraper uses category
    # pagination rather than this value.
    # -------------------------------------------------------------

    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help=(
            "Compatibility argument retained for "
            "run_pipeline.py. Category-based scraping "
            "uses category pagination."
        ),
    )

    parser.add_argument(
        "--offline-fixture",
        action="store_true",
        help=(
            "Use deterministic offline fixture "
            "instead of the live website."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help=(
            "Delay between category page requests "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--raw-out",
        default=(
            "data_pipeline/data/"
            "raw_books.csv"
        ),
        help="Path for raw scraped CSV.",
    )

    parser.add_argument(
        "--clean-out",
        default=(
            "data_pipeline/data/"
            "cleaned_books.csv"
        ),
        help="Path for cleaned CSV.",
    )

    parser.add_argument(
        "--report-out",
        default=(
            "data_pipeline/data/"
            "scrape_report.json"
        ),
        help="Path for JSON report.",
    )

    args = parser.parse_args()

    raw_path = Path(args.raw_out)
    clean_path = Path(args.clean_out)
    report_path = Path(args.report_out)

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Scraping
    # -------------------------------------------------------------

    if args.offline_fixture:

        print(
            "Using offline deterministic fixture..."
        )

        books = build_offline_fixture()

        scrape_stats = {
            "mode": "offline_fixture",
            "listing_pages_requested": 0,
            "book_links_seen": len(books),
            "unique_books": len(books),
            "detail_requests": 0,
            "failed_detail_requests": 0,
        }

    else:

        print(
            "Connecting to Books to Scrape..."
        )

        session = build_session()

        category_urls = (
            discover_category_urls(
                session
            )
        )

        print(
            "\nDiscovered target categories:"
        )

        for category, url in (
            category_urls.items()
        ):
            print(
                f"  {category}: {url}"
            )

        missing_categories = [
            category
            for category in TARGET_CATEGORIES
            if category not in category_urls
        ]

        if missing_categories:
            raise RuntimeError(
                "Could not discover the following "
                "required categories: "
                + ", ".join(
                    missing_categories
                )
            )

        books = scrape_categories(
            session=session,
            category_urls=category_urls,
            delay=max(
                0.0,
                args.delay,
            ),
        )

        unique_titles = {
            book["title"]
            for book in books
            if book.get("title")
        }

        scrape_stats = {
            "mode": "live",
            "listing_pages_requested": (
                None
            ),
            "book_links_seen": len(books),
            "unique_books": len(
                unique_titles
            ),
            "detail_requests": 0,
            "failed_detail_requests": 0,
            "categories_requested": (
                TARGET_CATEGORIES
            ),
        }

    # -------------------------------------------------------------
    # Save raw data
    # -------------------------------------------------------------

    raw_df = pd.DataFrame(books)

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    print(
        f"\nRaw data saved to: {raw_path}"
    )

    # -------------------------------------------------------------
    # Clean data
    # -------------------------------------------------------------

    cleaned_df, cleaning_report = (
        clean_books(books)
    )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_final_dataset(
        cleaned_df
    )

    # -------------------------------------------------------------
    # Save cleaned data
    # -------------------------------------------------------------

    cleaned_df.to_csv(
        clean_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------

    save_report(
        report_path=report_path,
        scrape_stats=scrape_stats,
        cleaning_report=cleaning_report,
        df=cleaned_df,
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print(
        f"Final rows       : {len(cleaned_df)}"
    )

    print(
        f"Categories       : "
        f"{cleaned_df['category'].nunique()}"
    )

    print(
        f"Fixed conversion : "
        f"1 GBP = {FIXED_GBP_TO_INR} INR"
    )

    print(
        "\nCategory counts:"
    )

    print(
        cleaned_df[
            "category"
        ].value_counts()
    )

    print(
        f"\nCleaned data saved to:"
        f"\n{clean_path}"
    )

    print(
        f"\nReport saved to:"
        f"\n{report_path}"
    )

    print(
        "\nCleaning report:"
    )

    print(
        json.dumps(
            cleaning_report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()