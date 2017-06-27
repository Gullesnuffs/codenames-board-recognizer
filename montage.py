import os
from subprocess import call
import sys

if len(sys.argv) < 3:
    print("Usage:\n\tpython3 montage.py script.py files...\n\tscript.py is assumed to write an output file called res.png")
    print("\nExample:\n\tpython3 montage.py grid.py examples/secret/*")
    exit(1)

call("rm -rf res", shell=True)
os.mkdir("res")
samples = []  # type: List[float]
cnt = 0
for file in sys.argv[2:]:
    call(["python3", sys.argv[1], file])
    os.rename("res.png", "res/" + os.path.basename(os.path.splitext(file)[0]) + ".png")
    cnt += 1

call("gm montage -geometry 300x -tile 5x -label \"%f\" res/*.png out.png", shell=True)
if sys.platform.startswith('linux'):
    call(["xdg-open", "out.png"])
elif sys.platform == 'darwin':
    call(["open", "out.png"])
else:
    os.startfile("out.png")
