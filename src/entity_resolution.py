import csv
import random
import re
import unicodedata
from collections import defaultdict, Counter


# ============================================================
# SETTINGS
# ============================================================

S1_FILE = "dataset/train/train_source1.tsv"
S2_FILE = "dataset/train/train_source2.tsv"
S3_FILE = "dataset/train/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

SAMPLE_SIZE = 20_000

# Maximum number of S1 records allowed inside one block.
# Very common blocks create too many candidates.
MAX_BLOCK_FREQUENCY = 50

# Character n-gram length
NGRAM_SIZE = 3

random.seed(42)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Unicode-aware basic normalization.

    We keep alphanumeric characters and convert everything else
    into spaces.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    result = []

    for ch in text:
        if ch.isalnum():
            result.append(ch)
        else:
            result.append(" ")

    return " ".join("".join(result).split())


# ============================================================
# LEGAL / GENERIC WORDS
# ============================================================

LEGAL_WORDS = {
    "private",
    "limited",
    "ltd",
    "llp",
    "pvt",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "llc",
    "plc",
    "the",
    "m",
    "ms",
    "mss",
    "sri",
    "shri",
}


# ============================================================
# NAME TOKENS
# ============================================================

def get_name_tokens(name):
    """
    Return useful business-name tokens.

    Very short and legal/generic tokens are removed.
    """

    tokens = normalize_text(name).split()

    result = []

    for token in tokens:

        if token in LEGAL_WORDS:
            continue

        if len(token) < 3:
            continue

        result.append(token)

    return result


# ============================================================
# ADDRESS TOKENS
# ============================================================

def get_address_tokens(address):
    """
    Return useful address tokens.

    Numeric tokens are especially useful because house/building
    numbers can survive name/address transformations.
    """

    tokens = normalize_text(address).split()

    result = []

    for token in tokens:

        if len(token) < 3:
            continue

        if any(ch.isdigit() for ch in token):
            result.append(token)
            continue

        if len(token) >= 5:
            result.append(token)

    return result


# ============================================================
# ADDRESS NUMBERS
# ============================================================

def get_address_numbers(address):
    """
    Extract numeric components from an address.

    Examples:

        "12 MG Road Bangalore" -> {"12"}

        "Plot 45/2 Main Street" -> {"45", "2"}
    """

    normalized = normalize_text(address)

    numbers = set()

    for token in normalized.split():

        if token.isdigit():
            numbers.add(token)

        else:
            parts = re.findall(r"\d+", token)

            for part in parts:
                if part:
                    numbers.add(part)

    return numbers


# ============================================================
# NAME CHARACTER NGRAMS
# ============================================================

def get_name_ngrams(name):
    """
    Character 3-grams from the compact business name.

    This can survive small spelling changes.

    Example:

        "apple"

        app
        ppl
        ple
    """

    normalized = normalize_text(name)

    compact = normalized.replace(" ", "")

    if len(compact) < NGRAM_SIZE:
        return set()

    grams = set()

    for i in range(len(compact) - NGRAM_SIZE + 1):

        gram = compact[i:i + NGRAM_SIZE]

        grams.add(gram)

    return grams


# ============================================================
# NAME SIGNATURES
# ============================================================

def get_name_signatures(name):
    """
    Generate additional exact blocking signatures.

    These are useful when word order changes.
    """

    tokens = get_name_tokens(name)

    if not tokens:
        return set()

    signatures = set()

    # First useful token
    signatures.add("FIRST|" + tokens[0])

    # Last useful token
    signatures.add("LAST|" + tokens[-1])

    # Sorted first two useful tokens
    if len(tokens) >= 2:

        first_two = sorted(tokens[:2])

        signatures.add(
            "PAIR|" + "|".join(first_two)
        )

    # Any adjacent token pair
    if len(tokens) >= 2:

        for i in range(len(tokens) - 1):

            pair = sorted(
                [
                    tokens[i],
                    tokens[i + 1]
                ]
            )

            signatures.add(
                "PAIR|" + "|".join(pair)
            )

    return signatures


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

print("=" * 70)
print("STEP 1: SAMPLING SOURCE 1")
print("=" * 70)

sample_s1 = []

total_s1 = 0

with open(
    S1_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        total_s1 += 1

        if len(sample_s1) < SAMPLE_SIZE:

            sample_s1.append(row)

        else:

            j = random.randint(
                0,
                total_s1 - 1
            )

            if j < SAMPLE_SIZE:
                sample_s1[j] = row

        if total_s1 % 500_000 == 0:

            print(
                f"  Scanned {total_s1:,} S1 records..."
            )


print()
print(f"Total S1 records: {total_s1:,}")
print(f"Sampled S1 records: {len(sample_s1):,}")


# ============================================================
# BLOCK INDEX STRUCTURE
# ============================================================

blocks = {
    "name_token": defaultdict(set),
    "address_token": defaultdict(set),
    "address_number": defaultdict(set),
    "name_signature": defaultdict(set),
    "name_ngram": defaultdict(set),
}


block_frequency = {
    "name_token": Counter(),
    "address_token": Counter(),
    "address_number": Counter(),
    "name_signature": Counter(),
    "name_ngram": Counter(),
}


# ============================================================
# STEP 2: BUILD BLOCKING INDEXES
# ============================================================

print()
print("=" * 70)
print("STEP 2: BUILDING MULTI-BLOCKING INDEXES")
print("=" * 70)


for row in sample_s1:

    s1_id = row["entity_id"]

    country = row["country"].strip().lower()

    # --------------------------------------------------------
    # NAME TOKEN BLOCK
    # --------------------------------------------------------

    name_tokens = get_name_tokens(
        row["business_name"]
    )

    for token in set(name_tokens):

        key = country + "|NAME|" + token

        blocks["name_token"][key].add(s1_id)

        block_frequency["name_token"][key] += 1

    # --------------------------------------------------------
    # ADDRESS TOKEN BLOCK
    # --------------------------------------------------------

    address_tokens = get_address_tokens(
        row["business_address"]
    )

    for token in set(address_tokens):

        key = country + "|ADDR|" + token

        blocks["address_token"][key].add(s1_id)

        block_frequency["address_token"][key] += 1

    # --------------------------------------------------------
    # ADDRESS NUMBER BLOCK
    # --------------------------------------------------------

    address_numbers = get_address_numbers(
        row["business_address"]
    )

    for number in address_numbers:

        key = country + "|NUM|" + number

        blocks["address_number"][key].add(s1_id)

        block_frequency["address_number"][key] += 1

    # --------------------------------------------------------
    # NAME SIGNATURE BLOCK
    # --------------------------------------------------------

    signatures = get_name_signatures(
        row["business_name"]
    )

    for signature in signatures:

        key = country + "|SIG|" + signature

        blocks["name_signature"][key].add(s1_id)

        block_frequency["name_signature"][key] += 1

    # --------------------------------------------------------
    # NAME CHARACTER NGRAM BLOCK
    # --------------------------------------------------------

    ngrams = get_name_ngrams(
        row["business_name"]
    )

    for gram in ngrams:

        key = country + "|NGRAM|" + gram

        blocks["name_ngram"][key].add(s1_id)

        block_frequency["name_ngram"][key] += 1


# ============================================================
# FILTER COMMON BLOCKS
# ============================================================

print()
print("Filtering common blocks...")


filtered_blocks = {}

for block_type in blocks:

    filtered = {}

    for key, ids in blocks[block_type].items():

        frequency = block_frequency[block_type][key]

        if frequency <= MAX_BLOCK_FREQUENCY:

            filtered[key] = ids

    filtered_blocks[block_type] = filtered

    print(
        f"{block_type:20s}: "
        f"{len(filtered):,} usable blocks"
    )


# ============================================================
# STEP 3: LOAD GROUND TRUTH
# ============================================================

print()
print("=" * 70)
print("STEP 3: LOADING GROUND TRUTH")
print("=" * 70)


sample_s1_ids = {
    row["entity_id"]
    for row in sample_s1
}


ground_truth = {}

true_match_to_s1 = defaultdict(set)

total_true_matches = 0


with open(
    GT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        s1_id = row["source1_entity_id"]

        if s1_id not in sample_s1_ids:
            continue

        value = row[
            "matched_entity_ids"
        ].strip()

        if value:

            matches = {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }

        else:

            matches = set()

        ground_truth[s1_id] = matches

        for source_id in matches:

            true_match_to_s1[
                source_id
            ].add(s1_id)

            total_true_matches += 1


print(
    f"Ground-truth rows: "
    f"{len(ground_truth):,}"
)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(row):

    country = row["country"].strip().lower()

    candidate_ids = set()

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    name_tokens = get_name_tokens(
        row["business_name"]
    )

    for token in set(name_tokens):

        key = country + "|NAME|" + token

        ids = filtered_blocks[
            "name_token"
        ].get(key)

        if ids:
            candidate_ids.update(ids)

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    address_tokens = get_address_tokens(
        row["business_address"]
    )

    for token in set(address_tokens):

        key = country + "|ADDR|" + token

        ids = filtered_blocks[
            "address_token"
        ].get(key)

        if ids:
            candidate_ids.update(ids)

    # --------------------------------------------------------
    # ADDRESS NUMBERS
    # --------------------------------------------------------

    address_numbers = get_address_numbers(
        row["business_address"]
    )

    for number in address_numbers:

        key = country + "|NUM|" + number

        ids = filtered_blocks[
            "address_number"
        ].get(key)

        if ids:
            candidate_ids.update(ids)

    # --------------------------------------------------------
    # NAME SIGNATURES
    # --------------------------------------------------------

    signatures = get_name_signatures(
        row["business_name"]
    )

    for signature in signatures:

        key = country + "|SIG|" + signature

        ids = filtered_blocks[
            "name_signature"
        ].get(key)

        if ids:
            candidate_ids.update(ids)

    # --------------------------------------------------------
    # NAME CHARACTER NGRAMS
    # --------------------------------------------------------

    ngrams = get_name_ngrams(
        row["business_name"]
    )

    for gram in ngrams:

        key = country + "|NGRAM|" + gram

        ids = filtered_blocks[
            "name_ngram"
        ].get(key)

        if ids:
            candidate_ids.update(ids)

    return candidate_ids


# ============================================================
# EVALUATION
# ============================================================

def evaluate_source(
    filename,
    source_name
):

    print()
    print("=" * 70)
    print(
        f"TESTING MULTI-BLOCKING - {source_name}"
    )
    print("=" * 70)

    found_matches = set()

    total_candidate_pairs = 0

    records_with_candidates = 0

    total_records = 0

    # Count how many true matches are recovered
    # by each broad blocking family.
    block_hits = Counter()

    with open(
        filename,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            total_records += 1

            candidate_ids = get_candidates(row)

            if candidate_ids:

                records_with_candidates += 1

                total_candidate_pairs += len(
                    candidate_ids
                )

            source_id = row["entity_id"]

            # ------------------------------------------------
            # TRUE MATCH CHECK
            # ------------------------------------------------

            if source_id in true_match_to_s1:

                true_s1_ids = true_match_to_s1[
                    source_id
                ]

                for s1_id in true_s1_ids:

                    if s1_id in candidate_ids:

                        found_matches.add(
                            (
                                s1_id,
                                source_id
                            )
                        )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if total_records % 1_000_000 == 0:

                print(
                    f"  Processed "
                    f"{total_records:,} records..."
                )

    # ========================================================
    # RESULTS
    # ========================================================

    recall = (
        len(found_matches)
        / total_true_matches
        if total_true_matches
        else 0
    )

    average_candidates = (
        total_candidate_pairs
        / len(sample_s1)
    )

    print()
    print(
        f"Records scanned: "
        f"{total_records:,}"
    )

    print(
        f"Records with candidates: "
        f"{records_with_candidates:,}"
    )

    print(
        f"True matches found: "
        f"{len(found_matches):,}"
    )

    print(
        f"Blocking recall: "
        f"{recall:.4%}"
    )

    print(
        f"Candidate pairs: "
        f"{total_candidate_pairs:,}"
    )

    print(
        f"Average candidates / S1: "
        f"{average_candidates:,.2f}"
    )

    return (
        recall,
        total_candidate_pairs,
        found_matches
    )


# ============================================================
# SOURCE 2
# ============================================================

s2_recall, s2_candidates, s2_found = evaluate_source(
    S2_FILE,
    "SOURCE 2"
)


# ============================================================
# SOURCE 3
# ============================================================

s3_recall, s3_candidates, s3_found = evaluate_source(
    S3_FILE,
    "SOURCE 3"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

combined_recall = (
    (s2_recall + s3_recall)
    / 2
)

combined_candidates = (
    s2_candidates
    + s3_candidates
)

combined_average = (
    combined_candidates
    / len(sample_s1)
)


print()
print()
print("=" * 70)
print("FINAL MULTI-BLOCKING RESULTS")
print("=" * 70)

print()

print(
    f"S2 blocking recall: "
    f"{s2_recall:.4%}"
)

print(
    f"S3 blocking recall: "
    f"{s3_recall:.4%}"
)

print(
    f"Average S2/S3 recall: "
    f"{combined_recall:.4%}"
)

print(
    f"Total candidate pairs: "
    f"{combined_candidates:,}"
)

print(
    f"Average candidates / S1: "
    f"{combined_average:,.2f}"
)

print()
print("=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)