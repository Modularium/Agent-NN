from core.model_context import ModelContext, TaskContext, AccessText
from core.privacy import AccessLevel
from core.privacy_filter import redact_context


def test_redact_context_basic():
    ctx = ModelContext(
        task_context=TaskContext(
            task_type="demo",
            description=AccessText(text="desc", access=AccessLevel.CONFIDENTIAL),
            input_data=AccessText(text="secret", access=AccessLevel.SENSITIVE),
        ),
        memory=[
            AccessText(text="ok", access=AccessLevel.PUBLIC),
            AccessText(text="hide", access=AccessLevel.CONFIDENTIAL),
        ],
    )
    redacted = redact_context(ctx, AccessLevel.INTERNAL)
    assert redacted.task_context.input_data.text == "[REDACTED]"
    assert redacted.memory[1].text == "[REDACTED]"
    assert redacted.metrics["context_redacted_fields"] == 2
