"""
Consolidated feature engineering module — Member 1, Review 3.

Imported by:
  - eda.py (training pipeline, this folder)
  - M3's Flask /predict endpoint (must import from here, not copy-paste)

The format of explain_row() output is FROZEN for M3 and M4.
"""
import re
import json
from urllib.parse import urlparse

import tldextract


# ============================================================
# CONSTANTS
# ============================================================
URGENCY_WORDS = {
    'urgent', 'immediately', 'verify', 'suspended', 'suspend', 'expire',
    'expires', 'expired', 'confirm', 'action', 'required', 'restricted',
    'locked', 'limited', 'alert', 'warning', 'act now', 'asap',
    'password', 'unauthorized', 'security', 'update your', 'click here',
    'validate', 'deactivate', 'final notice', 'reactivate',
}

URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'shorte.st', 'cutt.ly', 'rebrand.ly',
}

PLACEHOLDER_TOKENS = {
    'financialinfo', 'referencenumber', 'organization',
    'ipaddress', 'emailaddress',
}

COMMON_BRANDS = {
    'paypal', 'amazon', 'apple', 'microsoft', 'google', 'netflix',
    'bank', 'irs', 'facebook', 'instagram', 'chase', 'wellsfargo',
}

KNOWN_BRAND_DOMAINS = {
    'paypal': 'paypal.com',
    'amazon': 'amazon.com',
    'apple': 'apple.com',
    'microsoft': 'microsoft.com',
    'google': 'google.com',
    'netflix': 'netflix.com',
    'facebook': 'facebook.com',
    'instagram': 'instagram.com',
    'chase': 'chase.com',
    'wellsfargo': 'wellsfargo.com',
    'irs': 'irs.gov',
}

URL_PATTERN = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE)

FEATURE_COLS = [
    'link_count', 'urgency_density', 'placeholder_density',
    'sender_mismatch', 'has_ip_link', 'has_shortened_link',
    'has_punycode', 'lookalike_domain_score',
]

FEATURE_LABELS = {
    'link_count': 'contains multiple links',
    'urgency_density': 'urgent/pressuring language',
    'placeholder_density': 'requests personal/financial info',
    'sender_mismatch': 'sender/brand mismatch',
    'has_ip_link': 'link uses a raw IP address',
    'has_shortened_link': 'uses a shortened URL',
    'has_punycode': 'punycode/lookalike domain',
    'lookalike_domain_score': 'domain resembles a known brand',
}


# ============================================================
# TEXT CLEANING (used by placeholder_density only)
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# INDIVIDUAL FEATURE FUNCTIONS
# ============================================================
def link_count(text):
    return len(URL_PATTERN.findall(str(text)))


def urgency_density(text):
    text_lower = str(text).lower()
    word_count = max(len(text_lower.split()), 1)
    hits = sum(text_lower.count(w) for w in URGENCY_WORDS)
    return round((hits / word_count) * 100, 2)


def placeholder_density(cleaned_text):
    words = str(cleaned_text).split()
    word_count = max(len(words), 1)
    hits = sum(1 for w in words if w in PLACEHOLDER_TOKENS)
    return round((hits / word_count) * 100, 2)


def sender_mismatch(text):
    """Text-level proxy: brand mentioned but no matching link domain."""
    text_lower = str(text).lower()
    mentioned_brands = {b for b in COMMON_BRANDS if b in text_lower}
    if not mentioned_brands:
        return 0
    links = URL_PATTERN.findall(text_lower)
    if not links:
        return 1
    domain_hit = any(brand in link for brand in mentioned_brands for link in links)
    return 0 if domain_hit else 1


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def domain_features(text):
    """Runs on RAW text. Returns domain-level phishing indicators."""
    urls = URL_PATTERN.findall(str(text))
    feats = {
        'has_ip_link': 0,
        'has_shortened_link': 0,
        'has_punycode': 0,
        'lookalike_domain_score': 0.0,
    }
    for u in urls:
        # Python 3.14's urlparse raises ValueError on malformed URLs
        # (bad IPv6, stray brackets). Skip those instead of crashing.
        try:
            parsed = urlparse(u if u.startswith('http') else 'http://' + u)
            host = parsed.netloc.lower()
        except ValueError:
            continue

        host = host.split(':')[0]  # strip port
        if not host:
            continue

        if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host):
            feats['has_ip_link'] = 1

        if any(host == s or host.endswith('.' + s) for s in URL_SHORTENERS):
            feats['has_shortened_link'] = 1

        if 'xn--' in host:
            feats['has_punycode'] = 1

        try:
            ext = tldextract.extract(host)
        except Exception:
            continue
        reg = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        for brand, real in KNOWN_BRAND_DOMAINS.items():
            if reg and reg != real:
                d = _levenshtein(reg, real)
                if 0 < d <= 2 and len(reg) >= len(real) - 1:
                    score = 1.0 - d / max(len(real), 1)
                    feats['lookalike_domain_score'] = max(
                        feats['lookalike_domain_score'], round(score, 4)
                    )
    return feats


# ============================================================
# SINGLE ENTRY POINT — M3's /predict calls THIS
# ============================================================
def build_features(raw_text):
    """One raw email string -> dict of all 8 features."""
    cleaned = clean_text(raw_text)
    feats = {
        'link_count': link_count(raw_text),
        'urgency_density': urgency_density(raw_text),
        'placeholder_density': placeholder_density(cleaned),
        'sender_mismatch': sender_mismatch(raw_text),
    }
    feats.update(domain_features(raw_text))
    return feats


# ============================================================
# EXPLAINABILITY — frozen JSON format for M4
# ============================================================
def explain_row(row, means, stds, top_n=3):
    """
    row   : dict-like with all FEATURE_COLS keys
    means : dict of per-feature training means
    stds  : dict of per-feature training stds
    Returns a JSON string:
      {"triggers":[{"feature":..., "label":..., "weight":...}, ...]}
    """
    scores = {}
    for col in FEATURE_COLS:
        if row[col] > 0:
            z = (row[col] - means[col]) / stds[col] if stds[col] else 0.0
            scores[col] = float(z)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return json.dumps({
        "triggers": [
            {"feature": col, "label": FEATURE_LABELS[col], "weight": round(z, 3)}
            for col, z in top
        ]
    })


def save_explain_stats(means, stds, path):
    """Save training means/stds so the live API can reuse them."""
    with open(path, 'w') as f:
        json.dump({
            "means": {k: float(v) for k, v in means.items()},
            "stds":  {k: float(v) for k, v in stds.items()},
        }, f, indent=2)


def load_explain_stats(path):
    with open(path) as f:
        d = json.load(f)
    return d["means"], d["stds"]