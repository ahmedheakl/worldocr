from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

options = PdfPipelineOptions(generate_page_images=True, images_scale=300.0/72.0)

source = "pdf1.pdf" 
converter = DocumentConverter(format_options={
    "pdf": PdfFormatOption(pipeline_options=options)})
result = converter.convert(source)
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
    