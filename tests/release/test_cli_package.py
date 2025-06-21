import subprocess
import sys


def test_cli_entry_point():
    result = subprocess.run([sys.executable, "-m", "pip", "install", "."], capture_output=True)
    assert result.returncode == 0

    result = subprocess.run([sys.executable, "-m", "agentnn", "--version"], capture_output=True)
    assert result.returncode == 0
    assert b"1.0.0-beta" in result.stdout
