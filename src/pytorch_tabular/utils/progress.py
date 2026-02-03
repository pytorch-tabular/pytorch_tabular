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
            return lambda it, **kwargs: it
    elif backend == "tqdm":
        try:
            from tqdm.auto import tqdm
            return partial(tqdm, desc=description) if description else tqdm
        except ImportError:
            return lambda it, **kwargs: it
    else:  # none
        return lambda it, **kwargs: it


def get_progress_bar_callback(backend: str = "simple", enable_progress_bar: bool = True):
    """Get the appropriate PyTorch Lightning progress bar callback based on backend.

    Args:
        backend: The progress bar backend. Can be 'rich', 'simple', or 'none'.
        enable_progress_bar: Whether progress bar is enabled in trainer kwargs.

    Returns:
        A PyTorch Lightning callback or None.
    """
    if backend == "rich" and enable_progress_bar:
        try:
            from pytorch_lightning.callbacks import RichProgressBar
            return RichProgressBar()
        except ImportError:
            return None
    elif backend == "simple" and enable_progress_bar:
        try:
            from pytorch_lightning.callbacks import TQDMProgressBar
            return TQDMProgressBar()
        except ImportError:
            return None
    else:  # none
        return None