"""
Feature engineering for phishing email detection.

Computes three features for each row of a (potentially large) dataset:
  1. link_count               - number of hyperlinks in the email body
  2. urgency_keyword_density  - fraction of words that are "urgency" keywords
  3. sender_mismatch          - 1 if the display name / address pair looks
                                 spoofed, else 0

Designed for datasets too large to load fully into memory: reads and writes
in chunks via pandas' chunksize, so peak memory stays small regardless of
file size.

Usage:
    python extract_features.py
"""

import re
import csv
import pandas as pd
from email.utils import parseaddr
from urllib.parse import urlparse

# ----------------------------------------------------------------------
# CONFIG — matches this dataset's actual columns
# ----------------------------------------------------------------------
INPUT_PATH = r"C:\Users\Prometheus\Desktop\Project Exhibition I\dataset.csv"
OUTPUT_PATH = r"C:\Users\Prometheus\Desktop\Project Exhibition I\dataset_features.csv"
TEXT_COL = "body"                   # email body text
FROM_COL = "sender"                 # sender field (may or may not include a display name)
DOMAIN_COL = "sender_domain"        # dataset already provides this — no need to re-derive it
URL_COUNT_COL = "url_count"         # dataset already provides this — reused as link_count
CHUNK_SIZE = 50_000                 # rows per chunk; lower if memory-constrained
ENCODING = "utf-8"

# ----------------------------------------------------------------------
# 1. LINK COUNT
# ----------------------------------------------------------------------
# This dataset already ships a pre-computed `url_count` column, so there's
# no need to re-parse the body for links — we just carry that column over
# as `link_count` for naming consistency with the Review 2 slide. The
# regex-based extractor is kept below (unused by default) in case you ever
# need to compute it from raw text on a different dataset.
_URL_RE = re.compile(
    r"""(?:href\s*=\s*["'])?          # optional href=" prefix
        (?:https?://|www\.)           # scheme or www.
        [^\s"'<>]+                    # the rest of the URL
    """,
    re.IGNORECASE | re.VERBOSE,
)


def count_links(text: str) -> int:
    """Fallback: regex-based link count from raw text (not used by default)."""
    if not isinstance(text, str) or not text:
        return 0
    return len(_URL_RE.findall(text))


# ----------------------------------------------------------------------
# 2. URGENCY-KEYWORD DENSITY
# ----------------------------------------------------------------------
# Extend this list with whatever terms your EDA (Review 1) surfaced as
# high-signal. Kept lowercase; matching is case-insensitive.
URGENCY_KEYWORDS = {
    "urgent", "immediately", "immediate", "act now", "verify now",
    "verify your account", "suspended", "suspend", "expire", "expires",
    "expiring", "action required", "limited time", "final notice",
    "warning", "alert", "restricted", "confirm your identity",
    "unauthorized", "unusual activity", "click here", "password expired",
    "security alert", "your account will be", "failure to", "locked",
    "reactivate", "before it's too late",
}

# Pre-tokenize multi-word keywords so we can match phrases, not just words.
_PHRASE_KEYWORDS = sorted(
    (kw for kw in URGENCY_KEYWORDS if " " in kw), key=len, reverse=True
)
_WORD_KEYWORDS = {kw for kw in URGENCY_KEYWORDS if " " not in kw}

_WORD_RE = re.compile(r"\b[a-zA-Z']+\b")


def urgency_keyword_density(text: str) -> float:
    """
    Fraction of the email that consists of urgency language.
    Phrase matches count as one 'hit'; density = hits / total_word_count.
    Returns 0.0 for empty/invalid input rather than dividing by zero.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    lower = text.lower()
    words = _WORD_RE.findall(lower)
    total_words = len(words)
    if total_words == 0:
        return 0.0

    hits = 0
    remaining = lower
    for phrase in _PHRASE_KEYWORDS:
        hits += remaining.count(phrase)

    word_set_hits = sum(1 for w in words if w in _WORD_KEYWORDS)
    hits += word_set_hits

    return round(hits / total_words, 6)


# ----------------------------------------------------------------------
# 3. SENDER / DISPLAY-NAME MISMATCH
# ----------------------------------------------------------------------
# Optional: map brand names that commonly get impersonated to their real
# sending domain(s). Extend this as you see more brands in your dataset.
KNOWN_BRAND_DOMAINS = {
    "paypal": {"paypal.com"},
    "amazon": {"amazon.com"},
    "apple": {"apple.com", "icloud.com"},
    "microsoft": {"microsoft.com", "outlook.com", "live.com"},
    "google": {"google.com", "gmail.com"},
    "netflix": {"netflix.com"},
    "bank of america": {"bankofamerica.com"},
    "chase": {"chase.com"},
    "wells fargo": {"wellsfargo.com"},
    "irs": {"irs.gov"},
    "docusign": {"docusign.com", "docusign.net"},
}


def _domain_of(email_addr: str) -> str:
    if not email_addr or "@" not in email_addr:
        return ""
    return email_addr.rsplit("@", 1)[-1].strip().lower()


def sender_mismatch(from_field: str, known_domain: str = "") -> int:
    """
    Returns 1 if the sender looks spoofed, else 0. Two independent checks:

      a) Brand impersonation: display name mentions a known brand, but the
         sending domain doesn't belong to that brand's real domain set.

      b) Name/address inconsistency: display name itself contains an email
         address whose domain differs from the actual sending domain (a
         classic spoofing tell, e.g. "billing@real-bank.com" shown as the
         name while the real address is attacker@evil.com).

    `known_domain`: pass the dataset's own pre-computed sender_domain value
    when available — it's more reliable than re-deriving the domain from
    the raw `sender` field ourselves. Falls back to parsing `from_field`
    if `known_domain` is missing/blank.
    """
    if not isinstance(from_field, str) or not from_field.strip():
        return 0

    display_name, addr = parseaddr(from_field)
    display_lower = display_name.lower()

    domain = (known_domain or "").strip().lower()
    if not domain:
        domain = _domain_of(addr)

    # Check (a): brand mentioned in display name vs. actual sending domain
    for brand, real_domains in KNOWN_BRAND_DOMAINS.items():
        if brand in display_lower and domain not in real_domains:
            return 1

    # Check (b): an email address embedded in the display name that
    # disagrees with the real sending domain
    embedded_addr_match = re.search(r"[\w\.-]+@[\w\.-]+", display_name)
    if embedded_addr_match:
        embedded_domain = _domain_of(embedded_addr_match.group(0))
        if embedded_domain and embedded_domain != domain:
            return 1

    return 0


# ----------------------------------------------------------------------
# MAIN — chunked processing so large files never fully load into memory
# ----------------------------------------------------------------------
def main():
    first_chunk = True
    rows_processed = 0

    reader = pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
        encoding=ENCODING,
        engine="python",     # more tolerant of malformed rows than 'c'
        on_bad_lines="skip",
    )

    for chunk in reader:
        if TEXT_COL not in chunk.columns:
            raise KeyError(
                f"Column '{TEXT_COL}' not found. Available columns: {list(chunk.columns)}"
            )
        if FROM_COL not in chunk.columns:
            raise KeyError(
                f"Column '{FROM_COL}' not found. Available columns: {list(chunk.columns)}"
            )

        # Link count: reuse the dataset's existing url_count column if present,
        # otherwise fall back to regex-counting links in the body text.
        if URL_COUNT_COL in chunk.columns:
            chunk["link_count"] = chunk[URL_COUNT_COL]
        else:
            chunk["link_count"] = chunk[TEXT_COL].apply(count_links)

        chunk["urgency_keyword_density"] = chunk[TEXT_COL].apply(urgency_keyword_density)

        # Sender mismatch: pass the dataset's own sender_domain when available.
        if DOMAIN_COL in chunk.columns:
            chunk["sender_mismatch"] = chunk.apply(
                lambda row: sender_mismatch(row[FROM_COL], row[DOMAIN_COL]), axis=1
            )
        else:
            chunk["sender_mismatch"] = chunk[FROM_COL].apply(sender_mismatch)

        chunk.to_csv(
            OUTPUT_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )

        rows_processed += len(chunk)
        print(f"Processed {rows_processed:,} rows...")
        first_chunk = False

    print(f"Done. Wrote features for {rows_processed:,} rows to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()