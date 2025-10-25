from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from PIL import Image
import os
from glob import glob

options = PdfPipelineOptions(generate_page_images=True, images_scale=1)

sources = glob("data/omnidocbench_output_large/images/*.jpg")[:10]
converter = DocumentConverter(format_options={"image": ImageFormatOption(pipeline_options=options), "pdf": PdfFormatOption(options=options)})
results = converter.convert_all(sources)

for result in results:
    doc = result.document
    imgs_by_page = doc.get_visualization(
        show_label=True,              
        show_branch_numbering=False,  
        viz_mode="key_value",     
        show_cell_id=True            
    )
    for i, img in imgs_by_page.items():
        image_path = f"{doc.name}_{i}.jpg"
        img.save(image_path) 
        
    for i in doc.pages:
        doctag_path = f"{doc.name}_page_{i}.doctag.txt"
        markdown_path = f"{doc.name}_page_{i}.md"
        with open(doctag_path, "w", encoding="utf-8") as f:
            f.write(doc.export_to_doctags(pages=set([i])))

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(doc.export_to_markdown(page_no=i))
        
        page_image_path = f"{doc.name}_page_{i}_raw.jpg"
        doc.pages[i].image.pil_image.save(page_image_path)
    