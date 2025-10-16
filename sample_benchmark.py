import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# --------------------------------------------
# Step 1: Load and concatenate all parquet files
# --------------------------------------------
files = glob("data/train/*.parquet")
all_dfs = [pd.read_parquet(f) for f in tqdm(files)]
df = pd.concat(all_dfs, ignore_index=True)

vital_tags = {
    'equation': 10,
    'title': 2,
    'text': 1,
    'section_header_level_1': 2,
    'section_header_level_2': 2,
    'section_header_level_3': 2,
    'section_header_level_4': 2,
    'section_header_level_5': 2,
    'section_header_level_6': 2,
    'header': 3,
    'footer': 3,
    'quote': 2,
    'list': 2,
    'otsl': 5,
    'figure': 5,
    'table_caption': 5,
    'figure_caption': 5,
    'footnote': 5,
}

# TODO: 
# 1. remove table_header
# 2. remove form_tag
# 3. convert annotation to text
# 4. if the element before table_caption is a figure, convert the tag name from table_caption to figure_caption

# remove any <table_header><loc_{l1}><loc_{l2}>...<loc_{ln}></table_header> pattern from the from columns ['doctag_html', 'doctag_otsl']
# remove any <form_tag><loc_{l1}><loc_{l2}>...<loc_{ln}></form_tag> pattern from the ['doctag_html', 'doctag_otsl']
def clean_doctag_columns(doctag: str) -> str:
    import re

    # 1) remove blocks
    doctag = re.sub(r"(?is)<\s*table_header\b[^>]*>.*?</\s*table_header\s*>", "", doctag)
    doctag = re.sub(r"(?is)<\s*form_tag\b[^>]*>.*?</\s*form_tag\s*>", "", doctag)

    # 2) annotation -> text
    doctag = re.sub(r"(?i)<\s*annotation\s*>", "<text>", doctag)
    doctag = re.sub(r"(?i)</\s*annotation\s*>", "</text>", doctag)

    # 3) index structural tags (figure/table only)
    struct_tags = list(re.finditer(r"(?i)<\s*/?\s*(figure|table)\b[^>]*>", doctag))
    struct_pos = [(m.start(), m.group(1).lower()) for m in struct_tags]

    def nearest_struct_type(start_idx: int, end_idx: int):
        """Return 'figure', 'table', or None based on nearest structural tag around [start_idx, end_idx)."""
        import math
        best_type, best_dist = None, math.inf

        # nearest previous
        for pos, typ in reversed(struct_pos):
            if pos < start_idx:
                d = start_idx - pos
                best_type, best_dist = typ, d
                break

        # nearest next
        for pos, typ in struct_pos:
            if pos > end_idx:
                d = pos - end_idx
                if d < best_dist:
                    best_type, best_dist = typ, d
                break

        return best_type

    # 4) rename caption blocks based on nearest structure
    out, last = [], 0
    caption_pat = re.compile(r"(?is)<\s*table_caption\b([^>]*)>(.*?)</\s*table_caption\s*>")
    for m in caption_pat.finditer(doctag):
        s, e = m.span()
        attrs = m.group(1) or ""      # preserve any attrs if present
        inner = m.group(2)

        out.append(doctag[last:s])

        typ = nearest_struct_type(s, e)
        if typ == "figure":
            out.append(f"<figure_caption{attrs}>{inner}</figure_caption>")
        else:
            out.append(f"<table_caption{attrs}>{inner}</table_caption>")

        last = e

    out.append(doctag[last:])
    doctag = "".join(out)

    return doctag


df['doctag_html'] = df['doctag_html'].apply(clean_doctag_columns)
df['doctag_otsl'] = df['doctag_otsl'].apply(clean_doctag_columns)


def get_score(doctag_otsl: str):
    score = 0
    for tag, s in vital_tags.items():
        cnt = doctag_otsl.count(tag)
        score += cnt * s
    return score


# must have these columns: 'language' and 'difficulty_score'
assert {'language', 'difficulty_score'}.issubset(df.columns), "Missing columns!"

# --------------------------------------------
# Step 2: Function to sample 100 examples per language
# --------------------------------------------
def sample_normal_distribution(group: pd.DataFrame, n_samples: int = 100):
    # Sort by difficulty
    group = group.sort_values('difficulty_score').reset_index(drop=True)
    
    # Compute percentiles (approximate normal coverage)
    percentiles = np.linspace(0, 100, len(group))
    group['percentile'] = percentiles

    # Ideal percentiles for normal-like shape (more from middle)
    # 5 bins (tails smaller, center denser)
    target_bins = [0, 10, 30, 70, 90, 100]
    bin_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # approximate bell shape
    bin_counts = np.round(bin_weights * n_samples).astype(int)

    sampled = []
    for (low, high), count in zip(zip(target_bins[:-1], target_bins[1:]), bin_counts):
        candidates = group[(group['percentile'] >= low) & (group['percentile'] < high)]
        if len(candidates) > count:
            candidates = candidates.sample(count, random_state=42)
        sampled.append(candidates)
    return pd.concat(sampled)

# --------------------------------------------
# Step 3: Apply per-language sampling
# --------------------------------------------
df['difficulty_score'] = df['doctag_otsl'].apply(get_score)
benchmark_samples = df.groupby('language', group_keys=False).apply(sample_normal_distribution)

# --------------------------------------------
# Step 4: Save the benchmark
# --------------------------------------------
benchmark_samples = benchmark_samples.drop(columns=['percentile'], errors='ignore')
benchmark_samples.to_parquet("data/benchmark_100_per_lang_v3.parquet")

print("✅ Benchmark created with", len(benchmark_samples), "samples across",
      benchmark_samples['language'].nunique(), "languages.")
