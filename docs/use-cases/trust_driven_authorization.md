# Trust Driven Authorization

An agent may earn additional privileges when its trust score rises. After ten successful retrieval tasks a `retriever` could become an `analyst`.

Use `agentnn trust eligible <agent> --for analyst` to check if the requirements are met. If eligible, run `agentnn agent elevate <agent> --to analyst` which updates the contract.
