from argparse import ArgumentParser
from logging import INFO, basicConfig, DEBUG
from pathlib import Path
import sys

src_dir = (Path(__file__).resolve().parent.parent).resolve()
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))
    root_dir = src_dir.parent
    sys.path.append(str(root_dir))

parser = ArgumentParser(description="Generate a WAV file using the Spark-TTS model.")

parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("results"),
    help="Directory to save the generated outputs."
)
parser.add_argument(
    "--narrator-gender",
    type=str,
    default="male",
    choices=["male", "female"],
    help="Gender of the narrator."
)
parser.add_argument(
    "--narrator-pitch",
    type=str,
    default="very_low",
    choices=["very_low", "low", "moderate", "high", "very_high"],
    help="Pitch of the narrator."
)
parser.add_argument(
    "--narrator-speed",
    type=str,
    default="moderate",
    choices=["very_slow", "slow", "moderate", "fast", "very_fast"],
    help="Speed of the narrator."
)
parser.add_argument(
    "--narrator-path",
    type=int,
    default=None,
    help="Provide a path to narrator-produced WAV file to use as a base for the new narrator."
)
parser.add_argument(
    "--narrator-text",
    type=str,
    default=None,
    help="Text content to be used by the narrator. If not provided, narrator path will be used with .txt suffix"
)
parser.add_argument(
    "--narrator-seed",
    type=int,
    default=None,
    help="Seed for random number generation to initialize the voice model."
)
parser.add_argument(
    "--content",
    type=str,
    default="It was a dark and stormy night. Strange things were afoot.",
    help="Text content to be converted to speech."
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Enable verbose output."
)
args = parser.parse_args()

basicConfig(level=DEBUG if args.verbose else INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from pythonapi import generate_wav
from pythonapi.interfaces import Content, Narrator

generate_wav(
    content=Content(lines=[args.content]),
    narrator=Narrator(seed=args.narrator_seed, gender=args.narrator_gender, pitch=args.narrator_pitch, speed=args.narrator_speed),
    output_dir=Path("results")
)
