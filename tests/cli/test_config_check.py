from typer.testing import CliRunner
from sdk.cli.main import app


def test_config_check():
    runner = CliRunner()
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 0
    assert "DATA_DIR" in result.stdout
