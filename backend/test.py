from pathlib import Path
import os

PROJECT_ROOT = Path(__file__)
print(PROJECT_ROOT)

def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

this_dir = os.path.dirname(os.path.abspath(__file__))
print(this_dir)

parent_dir = os.path.dirname(this_dir)
print(parent_dir)
