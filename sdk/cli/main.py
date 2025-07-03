"""Modular command line interface for Agent-NN."""
from __future__ import annotations

import typer

from .commands.root import app as root_app
from .commands.session import session_app
from .commands.agent import agent_app
from .commands.tasks import register as register_tasks, task_app
from .commands.model import register as register_model
from .commands.config_cmd import register as register_config
from .commands.governance import register as register_governance
from .commands.agentctl import register as register_agentctl
from .commands.dispatch import register as register_dispatch
from .commands.context import context_app

app = typer.Typer()

# root commands
app.add_typer(root_app)

# sub command groups
app.add_typer(session_app, name="session")
app.add_typer(agent_app, name="agent")
app.add_typer(context_app, name="context")
register_tasks(app)
register_model(app)
register_config(app)
register_governance(app)

# additional command registrations
register_agentctl(agent_app)
register_dispatch(task_app)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
