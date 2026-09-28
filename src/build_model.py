import os
import re
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio


# ============================================================
# Paths
# ============================================================

TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"

TEST_S1 = "dataset/test/test_source1.tsv"
TEST_S2 = "dataset/test/test_source2.tsv"
TEST_S3 = "dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Common business-name/address punctuation
    value = re.sub(r"[^\w\s]", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def token_normalize(value):
    value = normalize_text(value)

    # Remove common legal suffixes.
    suffixes = {
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
        "company",
    }

    tokens = [
        token
        for token in value.split()
        if token not in suffixes
    ]

    return " ".join(tokens)


# ============================================================
# Load data
# ============================================================

print("Loading training data...")

s1 = pd.read_csv(TRAIN_S1, sep="\t")
s2 = pd.read_csv(TRAIN_S2, sep="\t")
s3 = pd.read_csv(TRAIN_S3, sep="\t")

ground_truth = pd.read_csv(
    GROUND_TRUTH,
    sep="\t"
)

print("Training data loaded.")

print("Source 1:", s1.shape)
print("Source 2:", s2.shape)
print("Source 3:", s3.shape)


# ============================================================
# Normalize
# ============================================================

print("\nNormalizing fields...")

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].apply(
        token_normalize
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_text
    )

    df["country_norm"] = df["country"].apply(
        normalize_text
    )

print("Normalization completed.")


# ============================================================
# Build blocking indexes
# ============================================================

print("\nBuilding blocking indexes...")

# Country + first characters of normalized name
#
# This avoids comparing millions of records against every
# other record.

def make_block_key(name, country):

    name = name.replace(" ", "")

    return (
        country,
        name[:3]
    )


s2["block_key"] = s2.apply(
    lambda row: make_block_key(
        row["name_norm"],
        row["country_norm"]
    ),
    axis=1
)

s3["block_key"] = s3.apply(
    lambda row: make_block_key(
        row["name_norm"],
        row["country_norm"]
    ),
    axis=1
)


s2_blocks = {}

for idx, row in s2.iterrows():

    key = row["block_key"]

    if key not in s2_blocks:
        s2_blocks[key] = []

    s2_blocks[key].append(idx)


s3_blocks = {}

for idx, row in s3.iterrows():

    key = row["block_key"]

    if key not in s3_blocks:
        s3_blocks[key] = []

    s3_blocks[key].append(idx)


print("Blocking indexes created.")


# ============================================================
# Matching function
# ============================================================

def calculate_score(source_row, candidate_row):

    name_score = token_set_ratio(
        source_row["name_norm"],
        candidate_row["name_norm"]
    )

    address_score = ratio(
        source_row["address_norm"],
        candidate_row["address_norm"]
    )

    country_score = (
        100
        if source_row["country_norm"]
        == candidate_row["country_norm"]
        else 0
    )

    # Conservative weighted score.
    #
    # Business name receives the highest weight because
    # it is generally the strongest identity signal.
    #
    # Address provides supporting evidence.
    #
    # Country is used as a consistency check.

    final_score = (
        0.60 * name_score
        + 0.35 * address_score
        + 0.05 * country_score
    )

    return final_score


# ============================================================
# Generate candidates for one source
# ============================================================

def generate_candidates(
    source1_row,
    target_df,
    block_index
):

    key = make_block_key(
        source1_row["name_norm"],
        source1_row["country_norm"]
    )

    indexes = block_index.get(key, [])

    candidates = []

    for idx in indexes:

        candidate = target_df.iloc[idx]

        score = calculate_score(
            source1_row,
            candidate
        )

        candidates.append(
            (
                candidate["entity_id"],
                score
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates


# ============================================================
# Test matching on first 100 Source 1 records
# ============================================================

print("\nTesting candidate generation...")

for _, row in s1.head(100).iterrows():

    s2_candidates = generate_candidates(
        row,
        s2,
        s2_blocks
    )

    s3_candidates = generate_candidates(
        row,
        s3,
        s3_blocks
    )

    all_candidates = (
        s2_candidates
        + s3_candidates
    )

    all_candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nSource 1:", row["entity_id"])
    print(
        "Name:",
        row["business_name"]
    )

    print("Top candidates:")

    for entity_id, score in all_candidates[:5]:

        print(
            " ",
            entity_id,
            "score=",
            round(score, 2)
        )


print("\nCandidate-generation test completed.")