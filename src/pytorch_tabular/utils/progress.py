"""Progress bar utilities for PyTorch Tabular."""

from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Iterator, Optional


class DummyProgress:
    """A dummy progress class that mimics rich.Progress but does nothing."""

    def add_task(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def track(self, iterable, *args, **kwargs):
        return iterable


def get_progress_tracker(backend: str = "none", description: Optional[str] = None) -> Callable[[Iterator], Iterator]:
    """Get a progress tracker function based on the backend.

    Args:
        backend: The progress bar backend. Can be 'rich', 'tqdm', or 'none'.
        description: Description for the progress bar.

    Returns:
        A function that takes an iterable and returns an iterator with progress tracking.
    """
    if backend == "rich":
        try:
            from rich.progress import track
            return partial(track, description=description) if description else track
        except ImportError:
            # Fallback to none if rich is not available
            return lambda it: it
    elif backend == "tqdm":
        try:
            from tqdm.auto import tqdm
            return partial(tqdm, desc=description) if description else tqdm
        except ImportError:
            return lambda it: it
    else:  # none
        return lambda it: it


def get_progress_context(backend: str = "none"):
    """Get a progress context manager based on the backend.

    Args:
        backend: The progress bar backend. Can be 'rich', 'tqdm', or 'none'.

    Returns:
        A context manager for progress tracking that has a track method.
    """
    if backend == "rich":
        try:
            from rich.progress import Progress
            return Progress()
        except ImportError:
            return DummyProgress()
    elif backend == "tqdm":
        # tqdm doesn't have a context manager like rich's Progress
        # For now, return DummyProgress
        return DummyProgress()
    else:
        return DummyProgress()