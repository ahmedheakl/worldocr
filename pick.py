import json
import shutil
import os
images_root = "data/omnidocbench_output_large/images"
viz_root = "data/omnidocbench_output_large/visualizations"
html_root = "data/omnidocbench_output_large/htmls"
doctags_root = "data/omnidocbench_output_large/doctags"
markdowns_root = "data/omnidocbench_output_large/markdowns"
outfolder = "sohail_db"
os.makedirs(outfolder, exist_ok=True)
os.makedirs(os.path.join(outfolder, "images"), exist_ok=True)
os.makedirs(os.path.join(outfolder, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(outfolder, "html"), exist_ok=True)
os.makedirs(os.path.join(outfolder, "doctags"), exist_ok=True)
os.makedirs(os.path.join(outfolder, "markdowns"), exist_ok=True)


with open("2col.json", "r") as f:
    files = json.load(f)
    
for image_path in files:
    filename = os.path.basename(image_path)
    filler = "_highres" if os.path.exists(os.path.join(viz_root, filename.replace(".jpg", "_highres.jpg"))) else ""
    if not os.path.exists(os.path.join(doctags_root, filename.replace(".jpg", f"{filler}.json"))): continue
    try:
        
        shutil.copy(image_path, os.path.join(outfolder, "images", filename))
        
        shutil.copy(os.path.join(viz_root, filename.replace(".jpg", f"{filler}.jpg")), os.path.join(outfolder, "visualizations", filename))
        shutil.copy(os.path.join(html_root, filename.replace(".jpg", f"{filler}.html")), os.path.join(outfolder, "html", filename.replace(".jpg", ".html")))
        shutil.copy(os.path.join(doctags_root, filename.replace(".jpg", f"{filler}.json")), os.path.join(outfolder, "doctags", filename.replace(".jpg", ".json")))
        shutil.copy(os.path.join(markdowns_root, filename.replace(".jpg", f"{filler}.md")), os.path.join(outfolder, "markdowns", filename.replace(".jpg", ".md")))
    except: continue