"""Inclusive CUDA allocator phases and weak ownership checks.

No CUDA is imported or initialized here. A caller supplies torch after admission.
The outer phase must cover every CUDA operation, including verification/cleanup.
Allocator measurements are neither total process VRAM nor NVML device memory.
"""
from contextlib import contextmanager
import time
import weakref


class Profiler:
    def __init__(self, torch, device=0):
        self.cuda, self.device = torch.cuda, device
        self.records, self.stack = [], []

    def snapshot(self, name=None, **context):
        self.cuda.synchronize(self.device)
        value = {"allocated_bytes": int(self.cuda.memory_allocated(self.device)),
                 "reserved_bytes": int(self.cuda.memory_reserved(self.device))}
        if name is not None:
            value.update(name=name, **context)
        return value

    def _propagate(self):
        peaks = {"peak_allocated_bytes": int(self.cuda.max_memory_allocated(self.device)),
                 "peak_reserved_bytes": int(self.cuda.max_memory_reserved(self.device))}
        for frame in self.stack:
            for key, value in peaks.items():
                frame[key] = max(frame[key], value)

    @contextmanager
    def phase(self, name, **context):
        before = self.snapshot()
        # Preserve peaks in the interval since the last child, before resetting.
        self._propagate()
        record = {**context, "phase_id": len(self.records), "name": name,
                  "parent_phase_id": self.stack[-1]["phase_id"] if self.stack else None,
                  "timing_semantics": "inclusive_nested_do_not_sum", "before": before,
                  "peak_allocated_bytes": before["allocated_bytes"],
                  "peak_reserved_bytes": before["reserved_bytes"], "completed": False}
        self.records.append(record)
        self.stack.append(record)
        self.cuda.reset_peak_memory_stats(self.device)
        begin = time.perf_counter()
        try:
            yield record
            record["completed"] = True
        except BaseException as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            try:
                after = self.snapshot()
                self._propagate()
                record.update(after=after, seconds=time.perf_counter() - begin)
            except BaseException as error:
                record["measurement_error"] = repr(error)
                record["completed"] = False
                raise
            finally:
                self.stack.pop()
            self.cuda.reset_peak_memory_stats(self.device)

    def measure(self, name, operation, **context):
        with self.phase(name, **context) as record:
            result = operation()
        return result, record


def weak_tensor_refs(named_tensors):
    """Retain only names and weak references; inventories must not be kept alive."""
    seen, references = set(), []
    for name, tensor in named_tensors:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            references.append((name, weakref.ref(tensor)))
    return references


def release_report(references):
    alive = [name for name, reference in references if reference() is not None]
    return {"tracked_tensor_objects": len(references), "alive_tensor_objects": len(alive),
            "alive_names": alive, "all_tracked_tensor_objects_released": not alive,
            "boundary": "Weak tensor-object ownership only; allocator remainder and process exit are separate observations."}
