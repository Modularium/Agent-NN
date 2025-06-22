# Efficient Team Assignment

When creating coalitions the dispatcher can now choose workers based on their
skills and running costs. The CLI exposes `agentnn agent top --metric cost` to
list the cheapest agents. Subtasks specify a required skill so that
`select_best_agent` picks a member with matching abilities and low load.
