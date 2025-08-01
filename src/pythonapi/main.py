from logging import basicConfig, DEBUG
from pathlib import Path
import sys

src_dir = (Path(__file__).resolve().parent.parent).resolve()
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))
    root_dir = src_dir.parent
    sys.path.append(str(root_dir))

basicConfig(level=DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from pythonapi import generate_wav
from pythonapi.interfaces import Content, Narrator

generate_wav(
    content=Content(lines=["It was a dark and stormy night."]),
    narrator=None,
    output_dir=Path("results")
)
