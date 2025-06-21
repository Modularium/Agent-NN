import subprocess
import sys


def test_sdk_import_version():
    from sdk import __version__
    assert __version__ == "1.0.0-beta"


def test_cli_version_option():
    result = subprocess.run([sys.executable, "scripts/agentnn", "--version"], capture_output=True)
    assert result.returncode == 0
    assert b"1.0.0-beta" in result.stdout
