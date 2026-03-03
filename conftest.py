"""
Pytest configuration file for project-wide settings and fixtures.
Ensures that 'src' and parent directories are in sys.path for imports.
"""

import sys
from pathlib import Path

# Add the project root to sys.path so 'src' and other modules can be imported
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
