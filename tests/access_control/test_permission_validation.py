from core.model_context import ModelContext, TaskContext, AccessText
from core.privacy_filter import filter_permissions


def test_filter_permissions_redacts_unallowed():
    ctx = ModelContext(
        task_context=TaskContext(
            task_type="demo",
            input_data=AccessText(text="secret", permissions=["writer"]),
        )
    )
    out = filter_permissions(ctx, "critic")
    assert out.task_context.input_data.text == "[REDACTED]"
    assert out.metrics["context_redacted_fields"] == 1
