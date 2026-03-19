import os
import re
import subprocess

slides_dir = r"c:\Users\edson.DESKTOP-54IJM4F\Documents\GitHub\TDI\slides"

env_list = ["align", "align*", "equation", "equation*", "eqnarray", "eqnarray*"]

for f in ["1 - Transmissores e modulações digitais.tex", "2 - Canal com ruído aditivo gaussiano branco (AWGN).tex"]:
    tex_path = os.path.join(slides_dir, f)
    if os.path.exists(tex_path):
        with open(tex_path, "r", encoding="utf-8") as file:
            content = file.read()
            
        content = content.replace(r"\large", "")
        
        # Unwrap standalone environments
        for env in env_list:
            content = content.replace(r"\[\begin{" + env + r"}", r"\begin{" + env + r"}")
            content = content.replace(r"\end{" + env + r"}\]", r"\end{" + env + r"}")
            content = re.sub(r'\\\[\s*\\begin\{' + env.replace('*', r'\*') + r'\}', r'\\begin{' + env + r'}', content)
            content = re.sub(r'\\end\{' + env.replace('*', r'\*') + r'\}\s*\\\]', r'\\end{' + env + r'}', content)

        # Remove blank lines inside \[ \] which cause "Display math" errors
        content = re.sub(r'\\\[(.*?)\\\]', lambda m: r'\[' + m.group(1).replace('\n\n', '\n').replace('\n\n\n', '\n') + r'\]', content, flags=re.DOTALL)

        with open(tex_path, "w", encoding="utf-8") as file:
            file.write(content)
            
        print(f"Fixed {f}. Compiling...")
        
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f], cwd=slides_dir)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f], cwd=slides_dir)

print("All compiled!")
