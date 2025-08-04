from dataclasses import asdict
from functools import lru_cache
from json import dump as json_dump
from json import loads as json_loads
from pathlib import Path

from pythonapi.interfaces import VoiceMetadata

class CannedVoice:
    """Voice details as JSON

    When voices wav is available, it is used to initialize the voice model
    and the details are stored in a JSON file.
    """
    def __init__(self, json_path: Path):
        self.json_path = json_path

    def read(self) -> VoiceMetadata:
        """Read voice metadata from JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Voice metadata file does not exist: {self.json_path}")
        with self.json_path.open('r', encoding='utf-8') as f:
            data = json_loads(f.read())
        return VoiceMetadata(**data)

    def write(self, voice_metadata: VoiceMetadata) -> None:
        """Write voice metadata to JSON file."""
        if not self.json_path.parent.exists():
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open('w', encoding='utf-8') as f:
            json_dump(asdict(voice_metadata), f, indent=4)

@lru_cache(maxsize=None)
def get_canned_voice(json_path: Path) -> VoiceMetadata:
    """Get a CannedVoice instance for the given path."""
    return CannedVoice(json_path).read()
