import json
import matplotlib.pyplot as plt

with open("data/omnidocbench_output_mega/omnidocbench.json", "r") as f:
    data = json.load(f)
    
languages = []
for sample in data:
    lang = sample['page_info']['page_attribute']['language']
    languages.append(lang)
    
lang_dist = {}
for lang in languages:
    if lang not in lang_dist:
        lang_dist[lang] = 0
    lang_dist[lang] += 1
sorted_lang_dist = dict(sorted(lang_dist.items(), key=lambda item: item[1], reverse=True))  
plt.figure(figsize=(12, 6))
plt.bar(sorted_lang_dist.keys(), sorted_lang_dist.values())
plt.xticks(rotation=90)
plt.xlabel("Language")
plt.ylabel("Number of Samples")
plt.title("Distribution of Languages in OmniDocBench Dataset")
plt.tight_layout()
plt.savefig("lang_distribution.png")