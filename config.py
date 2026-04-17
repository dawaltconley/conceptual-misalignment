from pathlib import Path

DIST = Path("dist")
DIST.mkdir(exist_ok=True)

TEXT = Path("text")
TEXT.mkdir(exist_ok=True)

SEP = TEXT / "sep"
SEP.mkdir(exist_ok=True)

CTEXT = TEXT / "ctext"
CTEXT.mkdir(exist_ok=True)
