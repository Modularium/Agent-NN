# Decision Log: Phase 5 & 6 Integration

During the implementation of the advanced learning and federation layers we reused parts of the deprecated
`distributed_training.py` module from `archive/legacy/nn_models_deprecated`.
The weight aggregation logic was adapted into the new `training/federated.py` module providing a light-weight
`FederatedAveraging` helper.

The federation manager was expanded with dynamic discovery and round‑robin scheduling based on ideas from
older prototypes in `archive/legacy`. Legacy service stubs were not kept.
