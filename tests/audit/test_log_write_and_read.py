from core.audit_log import AuditLog, AuditEntry
from datetime import datetime


def test_log_write_and_read(tmp_path):
    log = AuditLog(log_dir=tmp_path)
    entry = AuditEntry(
        timestamp=datetime.utcnow().isoformat(),
        actor="tester",
        action="unit_test",
        context_id="ctx1",
        detail={"ok": True},
    )
    log.write(entry)
    items = log.read_file(datetime.utcnow().date().isoformat())
    assert items[0]["action"] == "unit_test"
    assert log.by_context("ctx1")[0]["actor"] == "tester"
