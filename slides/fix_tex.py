import os
import re
import subprocess

slides_dir = r"c:\Users\edson.DESKTOP-54IJM4F\Documents\GitHub\TDI\slides"

for f in os.listdir(slides_dir):
    if f.endswith(".tex"):
        tex_path = os.path.join(slides_dir, f)
        with open(tex_path, "r", encoding="utf-8") as file:
            content = file.read()
            
        # Fix \large in math mode
        content = content.replace(r"\large", "")
        
        # Pandoc sometimes wraps \begin{align} with \[ \]
        content = content.replace(r"\[\begin{align}", r"\begin{align}")
        content = content.replace(r"\end{align}\]", r"\end{align}")
        
        # Also clean up \begin{eqnarray} if present
        content = content.replace(r"\[\begin{eqnarray}", r"\begin{eqnarray}")
        content = content.replace(r"\end{eqnarray}\]", r"\end{eqnarray}")

        # Sometimes pandas wraps the align* as well
        content = content.replace(r"\[\begin{align*}", r"\begin{align*}")
        content = content.replace(r"\end{align*}\]", r"\end{align*}")

        # And \begin{equation}
        content = content.replace(r"\[\begin{equation}", r"\begin{equation}")
        content = content.replace(r"\end{equation}\]", r"\end{equation}")
        content = content.replace(r"\[\begin{equation*}", r"\begin{equation*}")
        content = content.replace(r"\end{equation*}\]", r"\end{equation*}")
        
        # In case there's whitespace:
        content = re.sub(r'\\\[\s*\\begin\{equation\}', r'\\begin{equation}', content)
        content = re.sub(r'\\end\{equation\}\s*\\\]', r'\\end{equation}', content)
        content = re.sub(r'\\\[\s*\\begin\{equation\*\}', r'\\begin{equation*}', content)
        content = re.sub(r'\\end\{equation\*\}\s*\\\]', r'\\end{equation*}', content)

        with open(tex_path, "w", encoding="utf-8") as file:
            file.write(content)
            
        print(f"Fixed {f}. Compiling...")
        
        # Compile
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f], cwd=slides_dir)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f], cwd=slides_dir)

print("All fixed and compiled!")
