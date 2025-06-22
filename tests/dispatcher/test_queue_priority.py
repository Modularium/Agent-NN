from core.dispatch_queue import DispatchQueue
from core.model_context import ModelContext, TaskContext


def test_queue_priority_order():
    q = DispatchQueue()
    low = ModelContext(
        task="t1", task_context=TaskContext(task_type="demo"), priority=5
    )
    high = ModelContext(
        task="t2", task_context=TaskContext(task_type="demo"), priority=1
    )
    q.enqueue(low)
    q.enqueue(high)
    first = q.dequeue()
    assert first.priority == 1
