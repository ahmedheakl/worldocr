import json, csv, math, sys

path = "result/dotsocr_quick_match_metric_result.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Pull the language/group keys in a stable order
keys = sorted(data["text_block"]["group"]["Edit_dist"].keys())

top_header = []
header = []
row = []
cnt = 0
for k in keys:
    if "language" not in k: continue
    language = k.split(": text_")[-1]
    if "greek" in language: language = "greek"
    top_header.append(language)
    top_header.append(language)
    header.append("text")
    row.append(f"{data['text_block']['group']['Edit_dist'][k]:.3f}")

    header.append("table")
    teds = data["table"]["group"]["TEDS"].get(k, float("nan"))
    teds = round(float(teds) * 100.0, 2) 
    row.append("NaN" if (isinstance(teds, float) and math.isnan(teds)) else f"{teds:.2f}")
    cnt += 1
print(f"Processed {cnt} languages/groups")


with open("sheet_ready.tsv", "w", encoding="utf-8", newline="") as out:
    w = csv.writer(out, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(top_header)
    w.writerow(header)
    w.writerow(row)

print("\nWrote: sheet_ready.tsv")
