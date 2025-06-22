# Role Restricted Tasks

Some tasks contain sensitive data that should only be visible to certain agent roles. Input fields in a `TaskContext` may specify `permissions` with a list of allowed roles. When the dispatcher forwards the task to a worker, unauthorized fields are redacted. A reviewer, for example, can see the task description but not the confidential attachments which are only visible to writers.
