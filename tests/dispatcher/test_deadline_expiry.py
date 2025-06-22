from datetime import datetime, timedelta

from core.dispatch_queue import DispatchQueue
from core.model_context import ModelContext, TaskContext


def test_deadline_expiry():
    q = DispatchQueue()
    past = datetime.utcnow() - timedelta(minutes=1)
    future = datetime.utcnow() + timedelta(minutes=1)
    t1 = ModelContext(
        task="t1", task_context=TaskContext(task_type="demo"), deadline=past.isoformat()
    )
    t2 = ModelContext(
        task="t2",
        task_context=TaskContext(task_type="demo"),
        deadline=future.isoformat(),
    )
    q.enqueue(t1)
    q.enqueue(t2)
    expired = q.expire_old_tasks()
    assert expired[0].task == "t1"
    assert q.dequeue().task == "t2"
