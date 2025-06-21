# Interactive Session

An interactive writer can ask a critic for live feedback. The writer sends its
draft to the critic via the `AgentBus` and waits for a comment. After receiving
`feedback` it appends the suggestion and reports two iterations in its metrics.

This mechanism enables quick loops without leaving the current task context.
