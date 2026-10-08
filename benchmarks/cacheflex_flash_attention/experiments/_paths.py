"""Locate shared FA tools for historical scripts run from this directory."""
from pathlib import Path
import sys

FA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FA_DIR))
