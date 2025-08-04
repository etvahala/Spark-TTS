
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, List

@dataclass
class VoiceMetadata:
    """Voice details as JSON

    When voices wav is available, it is used to initialize the voice model
    and the following details are stored in a JSON file.
    """
    seed: int

    lines: List[str] = field(default_factory=list)

    input_directives: str = ""

class Constants:
    """
    Constants for the TTS system.
    """

    DEFAULT_NARRATOR_GENDER = "male"
    DEFAULT_NARRATOR_PITCH = "very_low"
    DEFAULT_NARRATOR_SPEED = "moderate"

    DEFAULT_TEMPERATURE = 0.8

    DEFAULT_TOKENS_PER_SEGMENT = 500  # Default value for tokens per segment if not specified

    DEFAULT_TOP_K = 50

    DEFAULT_TOP_P = 0.95

@dataclass
class Content:

    # Lines of text to be narrated
    lines: List[str] = field(default_factory=list)

    # Advanced: Optional segmentation threshold for tokenization, DEFAULT_TOKENS_PER_SEGMENT if not specified
    max_tokens_per_segment: Optional[int] = None

@dataclass
class PredefinedVoice:

    # Tokenized wav prompt file
    global_token_ids: Any

    # Transcript of the prompt audio
    prompt_as_text: str

@dataclass
class InputDirectives:
    """
    Input directives for the TTS model.
    """

    # Directives as text + placeholder for the content text to be narrated
    text: str

    # This is defined when the voice is generated from existing narrator voice
    predefined_voice: Optional[PredefinedVoice] = None

    # Seed for random number generation that was used to initialize the voice model
    seed: Optional[int] = None

    temperature: float = Constants.DEFAULT_TEMPERATURE

    top_k: int = Constants.DEFAULT_TOP_K

    top_p: float = Constants.DEFAULT_TOP_P

@dataclass
class NarratorVoiceSpec:

    gender: str = Constants.DEFAULT_NARRATOR_GENDER
    pitch: str = Constants.DEFAULT_NARRATOR_PITCH
    speed: str = Constants.DEFAULT_NARRATOR_SPEED


@dataclass
class Narrator:

    # Voice is either provided as parametrized spec, or as a path to a pregenerated voice JSON file (links to a WAV file)
    voice_spec: NarratorVoiceSpec | Path

    # Optional UInt32 seed for random number generation to initialize the voice model
    seed: Optional[int] = None

    # Control randomness in the TTS model (higher value: the more surprising the next token is)
    temperature: float = Constants.DEFAULT_TEMPERATURE

    # Smaller values limit the next token to the most likely ones (k limits the option to tokens with the highest probabilities)
    top_k: int = Constants.DEFAULT_TOP_K

    # Compared to top_k, top_p limits the next token to the most likely ones, but it does so by considering the smallest set of tokens with cumulative probability,
    # leading to a more flexible sampling strategy. When there are no large candidates, large p values allow for more diversity in the next token (more creativity).
    top_p: float = Constants.DEFAULT_TOP_P

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

    input_directives: InputDirectives

    output_file: Path | None = None
