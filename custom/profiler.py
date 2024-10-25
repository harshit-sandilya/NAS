import os
import torch
from pathlib import Path
from typing import Any, Dict
from pytorch_lightning.profilers import PyTorchProfiler

_KINETO_AVAILABLE = torch.profiler.kineto_available()


class Profiler(PyTorchProfiler):
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: str | None = None,
        record_module_names: bool = True,
        table_kwargs: Dict[str, Any] | None = None,
        **profiler_kwargs: Any,
    ) -> None:
        super().__init__(
            dirpath,
            filename,
            group_by_input_shapes,
            emit_nvtx,
            export_to_chrome,
            row_limit,
            sort_by_key,
            record_module_names,
            table_kwargs,
            **profiler_kwargs,
        )

    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if not self.function_events:
            return ""

        if self._export_to_chrome and not _KINETO_AVAILABLE:
            filename = f"{self.local_rank}_trace.json"
            path_to_trace = (
                filename
                if self.dirpath is None
                else os.path.join(self.dirpath, filename)
            )
            self.function_events.export_chrome_trace(path_to_trace)

        return None
