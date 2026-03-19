import json
import os
import re
import subprocess
import base64
import shutil

notebooks = [
    "1 - Transmissores e modulações digitais.ipynb",
    "2 - Canal com ruído aditivo gaussiano branco (AWGN).ipynb",
    "3 - Receptores ótimos para canais AWGN.ipynb",
    "4 - Transmissão digital em canais limitados em banda.ipynb",
    "5 - Equalização de canais de comunicação.ipynb"
]

output_dir = os.path.abspath(os.path.join("..", "slides"))
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "figuras"), exist_ok=True)

if os.path.exists("figuras"):
    for f in os.listdir("figuras"):
        src_path = os.path.join("figuras", f)
        if os.path.isfile(src_path):
            shutil.copy(src_path, os.path.join(output_dir, "figuras", f))

def create_markdown_for_pandoc(nb_path, nb_index):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    md_lines = []
    title = nb_path.replace(".ipynb", "")
    md_lines.append(f"---\nauthor: 'TDI Course'\ntitle: '{title}'\ntheme: 'Madrid'\n---\n")
    
    out_img_counter = 0
    
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            if '<h1>Table of Contents' in source or '<div class="toc">' in source:
                continue
                
            # Replace img tags pointing to figuras with markdown images
            source = re.sub(r'<img\s+src="[\./]*?(figuras/[^"]+)"[^>]*>', r'![](\1)', source)
            
            md_lines.append("\n" + source + "\n")
            
        elif cell["cell_type"] == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "display_data" and "image/png" in out.get("data", {}):
                    img_data = out["data"]["image/png"]
                    # USE SAFE ASCII NAME
                    img_filename = f"nb{nb_index}_out_{out_img_counter}.png"
                    out_img_counter += 1
                    
                    img_path = os.path.join(output_dir, "figuras", img_filename)
                    with open(img_path, "wb") as fimg:
                        fimg.write(base64.b64decode(img_data))
                    
                    md_lines.append(f"\n![](figuras/{img_filename})\n")
    
    temp_md_path = os.path.join(output_dir, title + ".md")
    with open(temp_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    return temp_md_path, title

for i, nb in enumerate(notebooks):
    if os.path.exists(nb):
        print(f"Processing {nb}...")
        temp_md, title = create_markdown_for_pandoc(nb, i)
        
        tex_path = os.path.join(output_dir, f"{title}.tex")
        # Run pandoc
        res_pandoc = subprocess.run(["pandoc", temp_md, "-t", "beamer", "-s", "--slide-level=2", "-o", tex_path], cwd=output_dir, capture_output=True, text=True)
        print("Pandoc output:", res_pandoc.stdout, res_pandoc.stderr)
        
        # FIX LATEX BEFORE COMPILING
        with open(tex_path, "r", encoding="utf-8") as ftex:
            content = ftex.read()
            
        content = content.replace(r"\large", "")
        content = content.replace(r"\[\begin{align}", r"\begin{align}")
        content = content.replace(r"\end{align}\]", r"\end{align}")
        content = content.replace(r"\[\begin{eqnarray}", r"\begin{eqnarray}")
        content = content.replace(r"\end{eqnarray}\]", r"\end{eqnarray}")
        content = content.replace(r"\[\begin{align*}", r"\begin{align*}")
        content = content.replace(r"\end{align*}\]", r"\end{align*}")
        content = content.replace(r"\[\begin{equation}", r"\begin{equation}")
        content = content.replace(r"\end{equation}\]", r"\end{equation}")
        content = re.sub(r'\\\[\s*\\begin\{equation\}', r'\\begin{equation}', content)
        content = re.sub(r'\\end\{equation\}\s*\\\]', r'\\end{equation}', content)

        with open(tex_path, "w", encoding="utf-8") as ftex:
            ftex.write(content)
            
        print("Fixed LaTeX output. Running pdflatex...")
        # Run pdflatex
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{title}.tex"], cwd=output_dir, capture_output=True, text=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{title}.tex"], cwd=output_dir, capture_output=True, text=True)
        
        print(f"Finished {nb}. PDF generated.")
    else:
        print(f"File {nb} not found.")

print("All done!")
