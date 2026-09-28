import csv
import os
import re
from collections import defaultdict
from rapidfuzz.fuzz import ratio, token_set_ratio

# ============================================================
# PATHS
# ============================================================

S1_PATH = "dataset/test/test_source1.tsv"
S2_PATH = "dataset/test/test_source2.tsv"
S3_PATH = "dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCHING_PATH = os.path.join(
    OUTPUT_DIR, "matching_results.tsv"
)

CANDIDATE_PATH = os.path.join(
    OUTPUT_DIR, "candidate_pairs.tsv"
)

# ============================================================
# SETTINGS
# ============================================================

# Maximum fuzzy candidates examined for one S1 record.
MAX_CANDIDATES = 50

# Minimum name similarity for fuzzy matching.
NAME_THRESHOLD = 80

# Final combined score.
FINAL_THRESHOLD = 88

# ============================================================
# NORMALIZATION
# ============================================================

SUFFIXES = {
    "private",
    "limited",
    "ltd",
    "pvt",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company"
}


def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_name(value):

    value = normalize_text(value)

    return " ".join(
        word
        for word in value.split()
        if word not in SUFFIXES
    )


def compact(value):

    return value.replace(" ", "")


# ============================================================
# BLOCKING KEYS
# ============================================================

def get_blocks(name, country):

    name = compact(name)

    if not name:
        return []

    blocks = []

    country = country or ""

    # Strong block
    blocks.append(
        country + "|" + name[:3]
    )

    # Stronger block
    if len(name) >= 4:
        blocks.append(
            country + "|" + name[:4]
        )

    # First two characters
    if len(name) >= 2:
        blocks.append(
            country + "|" + name[:2]
        )

    return blocks


# ============================================================
# STREAM TSV
# ============================================================

def read_tsv(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for row in reader:
            yield row


# ============================================================
# BUILD SOURCE 2 INDEX
# ============================================================

print("=" * 60)
print("FAST ENTITY RESOLUTION")
print("=" * 60)

print("\nLoading Source 2 index...")

s2_exact = defaultdict(list)
s2_blocks = defaultdict(list)

s2_count = 0

for row in read_tsv(S2_PATH):

    entity_id = row["entity_id"]

    name = normalize_name(
        row.get("business_name", "")
    )

    address = normalize_text(
        row.get("business_address", "")
    )

    country = normalize_text(
        row.get("country", "")
    )

    record = (
        entity_id,
        name,
        address,
        country
    )

    # Exact name + country index
    exact_key = (
        country + "|" + name
    )

    s2_exact[exact_key].append(record)

    # Blocking index
    for block in get_blocks(
        name,
        country
    ):

        s2_blocks[block].append(record)

    s2_count += 1

    if s2_count % 500000 == 0:

        print(
            f"Source 2 indexed: "
            f"{s2_count:,}"
        )


print(
    f"Source 2 complete: "
    f"{s2_count:,}"
)


# ============================================================
# BUILD SOURCE 3 INDEX
# ============================================================

print("\nLoading Source 3 index...")

s3_exact = defaultdict(list)
s3_blocks = defaultdict(list)

s3_count = 0

for row in read_tsv(S3_PATH):

    entity_id = row["entity_id"]

    name = normalize_name(
        row.get("business_name", "")
    )

    address = normalize_text(
        row.get("business_address", "")
    )

    country = normalize_text(
        row.get("country", "")
    )

    record = (
        entity_id,
        name,
        address,
        country
    )

    exact_key = (
        country + "|" + name
    )

    s3_exact[exact_key].append(record)

    for block in get_blocks(
        name,
        country
    ):

        s3_blocks[block].append(record)

    s3_count += 1

    if s3_count % 500000 == 0:

        print(
            f"Source 3 indexed: "
            f"{s3_count:,}"
        )


print(
    f"Source 3 complete: "
    f"{s3_count:,}"
)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if os.path.exists(MATCHING_PATH):
    os.remove(MATCHING_PATH)

if os.path.exists(CANDIDATE_PATH):
    os.remove(CANDIDATE_PATH)


# ============================================================
# OPEN OUTPUT FILES
# ============================================================

matching_file = open(
    MATCHING_PATH,
    "w",
    encoding="utf-8",
    newline=""
)

candidate_file = open(
    CANDIDATE_PATH,
    "w",
    encoding="utf-8",
    newline=""
)

matching_writer = csv.writer(
    matching_file,
    delimiter="\t"
)

candidate_writer = csv.writer(
    candidate_file,
    delimiter="\t"
)


# EXACT HEADERS REQUIRED BY VALIDATOR

matching_writer.writerow([
    "source1_entity_id",
    "matched_entity_ids"
])

candidate_writer.writerow([
    "source1_entity_id",
    "candidate_entity_ids"
])


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print("\nProcessing Source 1...")

processed = 0
matched = 0
candidate_rows = 0

for row in read_tsv(S1_PATH):

    source_id = row["entity_id"]

    source_name = normalize_name(
        row.get("business_name", "")
    )

    source_address = normalize_text(
        row.get("business_address", "")
    )

    source_country = normalize_text(
        row.get("country", "")
    )

    exact_key = (
        source_country + "|" + source_name
    )

    matches = {}

    # ========================================================
    # STEP 1 — EXACT NAME MATCH
    # ========================================================

    exact_candidates = []

    exact_candidates.extend(
        s2_exact.get(exact_key, [])
    )

    exact_candidates.extend(
        s3_exact.get(exact_key, [])
    )

    for candidate in exact_candidates:

        candidate_id = candidate[0]

        candidate_address = candidate[2]

        address_score = ratio(
            source_address,
            candidate_address
        )

        # Exact normalized business name is already strong.
        score = (
            0.75 * 100
            + 0.25 * address_score
        )

        matches[candidate_id] = score


    # ========================================================
    # STEP 2 — BLOCKING
    # ========================================================

    if not matches:

        candidate_pool = {}

        for block in get_blocks(
            source_name,
            source_country
        ):

            for candidate in s2_blocks.get(
                block,
                []
            ):

                candidate_pool[
                    candidate[0]
                ] = candidate

            for candidate in s3_blocks.get(
                block,
                []
            ):

                candidate_pool[
                    candidate[0]
                ] = candidate


        # ====================================================
        # Limit candidates for speed
        # ====================================================

        candidate_items = list(
            candidate_pool.values()
        )

        if len(candidate_items) > MAX_CANDIDATES:

            candidate_items = (
                candidate_items[
                    :MAX_CANDIDATES
                ]
            )


        # ====================================================
        # FUZZY MATCH
        # ====================================================

        for candidate in candidate_items:

            candidate_id = candidate[0]

            candidate_name = candidate[1]

            candidate_address = candidate[2]

            candidate_country = candidate[3]

            # Country mismatch
            if (
                source_country
                and candidate_country
                and source_country
                != candidate_country
            ):
                continue

            name_score = token_set_ratio(
                source_name,
                candidate_name
            )

            if name_score < NAME_THRESHOLD:
                continue

            address_score = ratio(
                source_address,
                candidate_address
            )

            score = (
                0.65 * name_score
                + 0.35 * address_score
            )

            if score >= FINAL_THRESHOLD:

                matches[
                    candidate_id
                ] = score


    # ========================================================
    # SORT MATCHES
    # ========================================================

    sorted_matches = sorted(
        matches.items(),
        key=lambda x: x[1],
        reverse=True
    )

    matched_ids = [
        item[0]
        for item in sorted_matches
    ]


    # ========================================================
    # WRITE MATCHING RESULT
    # ========================================================

    matching_writer.writerow([
        source_id,
        ",".join(matched_ids)
    ])


    # ========================================================
    # WRITE ONE CANDIDATE ROW PER S1
    # ========================================================

    candidate_writer.writerow([
        source_id,
        ",".join(matched_ids)
    ])


    processed += 1
    candidate_rows += 1

    if matched_ids:
        matched += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if processed % 10000 == 0:

        matching_file.flush()
        candidate_file.flush()

        print(
            f"Processed "
            f"{processed:,} / 1,732,544 | "
            f"Matched: {matched:,}"
        )


# ============================================================
# CLOSE FILES
# ============================================================

matching_file.close()
candidate_file.close()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("SUBMISSION GENERATION COMPLETE")
print("=" * 60)

print(
    f"Source 1 processed: "
    f"{processed:,}"
)

print(
    f"Source 1 matched: "
    f"{matched:,}"
)

print(
    f"Candidate rows: "
    f"{candidate_rows:,}"
)

print()
print(
    "Matching file:"
)

print(
    MATCHING_PATH
)

print()
print(
    "Candidate file:"
)

print(
    CANDIDATE_PATH
)

print("=" * 60)