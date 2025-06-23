# Frontend UX Checklist

This checklist summarises user experience and accessibility tests for the consolidated UI in `frontend/agent-ui`.

| Test Case | Criteria | Result |
|-----------|---------|-------|
| Keyboard navigation through sidebar | All links reachable via `Tab` key | ✅ |
| Chat send button | Shows loader and error state | ✅ |
| Feedback form validation | Empty input blocked, success message visible | ✅ |
| Responsive layout | Sidebar collapses on screens `<640px` | ✅ |
| ARIA attributes | Inputs and buttons include `aria-label` where needed | ✅ |
| `aria-live` regions | New chat messages announced to screen readers | ✅ |

Phase 3.3 addresses these items and the interface meets WCAG 2.1 AA contrast and focus requirements.
