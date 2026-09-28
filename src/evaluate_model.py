import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio
import re


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):

    value = normalize_text(value)

    suffixes = {
        "private", "limited", "ltd", "pvt",
        "llc", "inc", "incorporated",
        "corp", "corporation", "co", "company"
    }

    return " ".join(
        x for x in value.split()
        if x not in suffixes
    )


print("Loading training data...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Loaded.")


# Normalize
for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].apply(
        normalize_name
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_text
    )

    df["country_norm"] = df["country"].apply(
        normalize_text
    )


# Create ID lookups
s1_lookup = s1.set_index("entity_id").to_dict("index")
s2_lookup = s2.set_index("entity_id").to_dict("index")
s3_lookup = s3.set_index("entity_id").to_dict("index")


# Evaluate exact name/address evidence
total_matches = 0
name_exact = 0
address_exact = 0
both_exact = 0

print("Evaluating first 10,000 labelled Source-1 records...")

for _, row in truth.head(10000).iterrows():

    source1_id = row["source1_entity_id"]

    source1_row = s1_lookup.get(source1_id)

    if source1_row is None:
        continue

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            target = s2_lookup.get(matched_id)
        elif matched_id.startswith("S3-"):
            target = s3_lookup.get(matched_id)
        else:
            target = None

        if target is None:
            continue

        total_matches += 1

        name_score = token_set_ratio(
            source1_row["name_norm"],
            target["name_norm"]
        )

        address_score = ratio(
            source1_row["address_norm"],
            target["address_norm"]
        )

        if name_score >= 95:
            name_exact += 1

        if address_score >= 95:
            address_exact += 1

        if name_score >= 95 and address_score >= 95:
            both_exact += 1


print()
print("=" * 45)
print("MATCH EVIDENCE ANALYSIS")
print("=" * 45)

print("Ground-truth matches checked:", total_matches)

if total_matches:

    print(
        f"Name similarity >= 95: "
        f"{name_exact / total_matches * 100:.2f}%"
    )

    print(
        f"Address similarity >= 95: "
        f"{address_exact / total_matches * 100:.2f}%"
    )

    print(
        f"Both >= 95: "
        f"{both_exact / total_matches * 100:.2f}%"
    )

print("=" * 45)