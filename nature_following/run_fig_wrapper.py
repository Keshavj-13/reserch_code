import sys
import traceback
try:
    with open('notebooks/Generate_Manuscript_Figures.py', 'r', encoding='utf-8') as f:
        code = f.read()
    exec(code, globals())
except Exception as e:
    with open('figure_error.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    print("Caught exception, written to figure_error.txt")
