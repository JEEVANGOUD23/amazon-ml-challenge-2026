import pandas as pd
from rapidfuzz.fuzz import ratio


# ==============================
# Load training data
# ==============================

print("Loading training data...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Training data loaded!")


# ==============================
# Normalization
# ==============================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    for char in [",", ".", "-", "/", "(", ")", "&"]:
        value = value.replace(char, " ")

    return " ".join(value.split())


for df in [source1, source2, source3]:

    df["name_norm"] = df["business_name"].apply(
        normalize_text
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_text
    )


# ==============================
# Create ID lookup dictionaries
# ==============================

print("Creating lookup tables...")

source2_lookup = source2.set_index("entity_id").to_dict("index")
source3_lookup = source3.set_index("entity_id").to_dict("index")

print("Lookup tables created!")


# ==============================
# Analyze first 1000 labelled records
# ==============================

print("\nAnalyzing training matches...")

checked = 0
name_exact = 0
address_exact = 0
name_and_address_exact = 0

for _, row in ground_truth.head(1000).iterrows():

    source1_id = row["source1_entity_id"]

    source1_row = source1[
        source1["entity_id"] == source1_id
    ]

    if source1_row.empty:
        continue

    source1_row = source1_row.iloc[0]

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        if matched_id.startswith("S2-"):
            matched_row = source2_lookup.get(matched_id)

        elif matched_id.startswith("S3-"):
            matched_row = source3_lookup.get(matched_id)

        else:
            continue

        if matched_row is None:
            continue

        checked += 1

        same_name = (
            source1_row["name_norm"]
            == matched_row["name_norm"]
        )

        same_address = (
            source1_row["address_norm"]
            == matched_row["address_norm"]
        )

        if same_name:
            name_exact += 1

        if same_address:
            address_exact += 1

        if same_name and same_address:
            name_and_address_exact += 1


# ==============================
# Results
# ==============================

print("\n==============================")
print("TRAINING MATCH ANALYSIS")
print("==============================")

print("Matched records checked:", checked)

if checked > 0:

    print(
        "Exact name matches:",
        name_exact,
        f"({name_exact / checked * 100:.2f}%)"
    )

    print(
        "Exact address matches:",
        address_exact,
        f"({address_exact / checked * 100:.2f}%)"
    )

    print(
        "Exact name + address matches:",
        name_and_address_exact,
        f"({name_and_address_exact / checked * 100:.2f}%)"
    )

print("\nAnalysis completed!")