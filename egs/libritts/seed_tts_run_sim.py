#!/usr/bin/env python3
"""Run Seed-TTS SIM without shadowing Python's stdlib ``select`` module."""

import runpy
import select  # noqa: F401 -- preload the stdlib extension before adding SV_DIR
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: seed_tts_run_sim.py VERIFICATION_SCRIPT [arguments ...]")
    verification_script = Path(sys.argv[1]).expanduser().resolve()
    if not verification_script.is_file():
        raise FileNotFoundError(verification_script)

    # verification_pair_list_v2.py imports sibling modules as top-level names.
    # Add its directory only after stdlib select has been loaded and cached;
    # that directory also contains an unrelated select.py which would otherwise
    # break subprocess/socket imports with a circular-import error.
    sys.path.insert(0, str(verification_script.parent))
    sys.argv = [str(verification_script), *sys.argv[2:]]
    runpy.run_path(str(verification_script), run_name="__main__")


if __name__ == "__main__":
    main()
