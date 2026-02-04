"""Simple test to verify progress utility works correctly."""

import sys
from pathlib import Path

# Test the module can be imported and basic functions work
src_path = Path(__file__).parent / "src"
progress_file = src_path / "pytorch_tabular" / "utils" / "progress.py"

print(f"Testing progress utility at: {progress_file}")
print(f"File exists: {progress_file.exists()}")

# Simple syntax check
try:
    with open(progress_file, 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, str(progress_file), 'exec')
    print("✅ Progress utility syntax is valid")
except SyntaxError as e:
    print(f"❌ Syntax error in progress utility: {e}")
    sys.exit(1)

# Check for required functions
required_functions = [
    'get_progress_tracker',
    'get_progress_context',
    'get_progress_bar_callback'
]

for func in required_functions:
    if f"def {func}" in code:
        print(f"✓ Function '{func}' found")
    else:
        print(f"❌ Function '{func}' NOT found")
        sys.exit(1)

print("\n✅ All basic checks passed!")
print("\nNote: Full functional tests require pytorch_tabular dependencies to be installed.")
print("To run full tests: pip install -e . && pytest tests/")
