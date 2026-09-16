import json

def main():
    try:
        with open('src/lab1.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        with open('src/cells_summary.txt', 'w', encoding='utf-8') as out:
            for i, cell in enumerate(nb.get('cells', [])):
                if cell['cell_type'] == 'code':
                    source = cell.get('source', [])
                    if source:
                        first_line = source[0].strip()
                        out.write(f"Code Cell {i}:\n{first_line}\n...\n")
                    else:
                        out.write(f"Code Cell {i}: [EMPTY]\n...\n")
                elif cell['cell_type'] == 'markdown':
                    source = cell.get('source', [])
                    if source:
                        first_line = source[0].strip()
                        out.write(f"Markdown Cell {i}:\n{first_line}\n...\n")
                    else:
                        out.write(f"Markdown Cell {i}: [EMPTY]\n...\n")
    except Exception as e:
        with open('src/cells_summary.txt', 'w', encoding='utf-8') as out:
            out.write(str(e))

if __name__ == '__main__':
    main()
