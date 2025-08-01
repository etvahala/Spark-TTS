
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, List

class Constants:
    """
    Constants for the TTS system.
    """

    DEFAULT_NARRATOR_GENDER = "male"
    DEFAULT_NARRATOR_PITCH = "moderate"
    DEFAULT_NARRATOR_SPEED = "moderate"

    DEFAULT_TOKENS_PER_SEGMENT = 500  # Default value for tokens per segment if not specified

@dataclass
class Narrator:

    gender: str = Constants.DEFAULT_NARRATOR_GENDER
    pitch: str = Constants.DEFAULT_NARRATOR_PITCH
    speed: str = Constants.DEFAULT_NARRATOR_SPEED

    # Optional UInt32 seed for random number generation to initialize the voice model
    seed: Optional[int] = None

@dataclass
class Content:

    # Lines of text to be narrated
    lines: List[str] = field(default_factory=list)

    # Advanced: Optional segmentation threshold for tokenization, DEFAULT_TOKENS_PER_SEGMENT if not specified
    max_tokens_per_segment: Optional[int] = None

@dataclass
class OutputSpec:
    """
    Specification for the output of the TTS system.
    """
    output_dir: Path

@dataclass
class TokenizedContent:
    """
    Tokenized content for TTS processing.
    """
    segment_iterator: Iterator[str]

    input_directives: str = ""
