import sys
from pathlib import Path
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Initial sys.path: {sys.path[:3]}")

# Simulation of the notebook logic
IN_COLAB = False

# Add src to path
if IN_COLAB:
    sys.path.insert(0, str(Path.cwd()))
else:
    # If running locally in notebooks/, add parent directory
    # Try to find src
    current_dir = Path.cwd()
    if (current_dir / 'src').exists():
            print("Found src in current directory, adding current directory to path")
            sys.path.insert(0, str(current_dir))
    elif (current_dir.parent / 'src').exists():
            print("Found src in parent directory, adding parent directory to path")
            sys.path.insert(0, str(current_dir.parent))
    else:
            print("Warning: Could not find src directory")

print(f"Updated sys.path: {sys.path[:3]}")

try:
    import src.data
    print("Success: Imported src.data")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
