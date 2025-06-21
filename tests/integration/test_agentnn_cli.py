import subprocess
import sys


def test_agentnn_help():
    result = subprocess.run([sys.executable, "scripts/agentnn", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"submit" in result.stdout
