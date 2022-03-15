import torch
import torch.jit
import torch.nn as nn


def fill_unsupported(module: nn.Module, *names: str):
    """
    Fill unsupported method names on the module with a function that raises an
    exception when it is called. This can be used to appease TorchScript.
    """
    for name in names:
        if not hasattr(module, name):

            @torch.jit.export
            def unsupported_fn(
                self, _: torch.Tensor, unsup_method_name: str = name
            ) -> torch.Tensor:
                raise RuntimeError(f"method {unsup_method_name} is not supported on this object")

            setattr(module, name, unsupported_fn)
