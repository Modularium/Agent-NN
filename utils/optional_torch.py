try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False

    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchStub:
        cuda = _CudaStub()

        def __getattr__(self, name):
            raise ImportError("torch is not installed")

    torch = _TorchStub()  # type: ignore

__all__ = ["torch", "TORCH_AVAILABLE"]
