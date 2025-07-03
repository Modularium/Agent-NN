# Session Snapshots

Snapshots store the full conversation context and metadata for a session. They
can be created at any time and restored later to reproduce a run.

```
from agentnn.storage import snapshot_store
snap_id = snapshot_store.save_snapshot(session_id)
new_session = snapshot_store.restore_snapshot(snap_id)
```

Snapshots are saved as JSON files under `data/snapshots/`.
