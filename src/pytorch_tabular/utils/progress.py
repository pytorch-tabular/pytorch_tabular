# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Progress Bar Utilities - Abstraction layer for different progress bar backends."""

from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Callable, Iterator, Optional, Union

from pytorch_tabular.utils import get_logger

logger = get_logger(__name__)


def get_progress_tracker(
    backend: str = "none",
    iterable: Optional[Any] = None,
    description: Optional[str] = None,
    total: Optional[int] = None,
    **kwargs,
) -> Union[Iterator, Callable]:
    """Get a progress tracker for iterating over an iterable with progress display.

    This function provides a unified interface for different progress bar backends
    (rich, tqdm, or none). It abstracts away the specific implementation details
    of each backend.

    Args:
        backend (str): Progress bar backend to use. One of ['rich', 'tqdm', 'none', 'simple'].
            Defaults to 'none'.
        iterable (Optional[Any]): The iterable to track. If None, returns a callable that
            can be used to wrap an iterable later.
        description (Optional[str]): Description text to show with the progress bar.
        total (Optional[int]): Total number of iterations (if iterable doesn't have __len__).
        **kwargs: Additional keyword arguments to pass to the backend progress bar.

    Returns:
        Union[Iterator, Callable]: Either an iterator with progress tracking or the original
            iterable if backend is 'none'.

    Example:
        >>> # Track a simple iteration
        >>> for item in get_progress_tracker('tqdm', my_list, description='Processing'):
        ...     process(item)
        >>>
        >>> # Use as a wrapper function
        >>> track = get_progress_tracker('rich', description='Processing')
        >>> for item in track(my_list):
        ...     process(item)
    """
    # Normalize backend names
    if backend in ("simple", "none", None):
        backend = "none"
    elif backend not in ("rich", "tqdm"):
        logger.warning(f"Unknown progress bar backend '{backend}'. Defaulting to 'none'.")
        backend = "none"

    if backend == "none":
        # No progress bar - return iterable as-is or identity function
        if iterable is not None:
            return iterable
        else:
            return lambda it, **kw: it

    elif backend == "rich":
        try:
            from rich.progress import track

            if iterable is not None:
                return track(iterable, description=description, total=total, **kwargs)
            else:
                return partial(track, description=description, total=total, **kwargs)
        except ImportError:
            logger.warning(
                "rich is not installed. Install it with 'pip install rich' to use rich progress bars. "
                "Falling back to no progress bar."
            )
            if iterable is not None:
                return iterable
            else:
                return lambda it, **kw: it

    elif backend == "tqdm":
        try:
            from tqdm.auto import tqdm

            if iterable is not None:
                return tqdm(iterable, desc=description, total=total, **kwargs)
            else:
                return partial(tqdm, desc=description, total=total, **kwargs)
        except ImportError:
            logger.warning(
                "tqdm is not installed. Install it with 'pip install tqdm' to use tqdm progress bars. "
                "Falling back to no progress bar."
            )
            if iterable is not None:
                return iterable
            else:
                return lambda it, **kw: it


@contextmanager
def get_progress_context(backend: str = "none", **kwargs):
    """Get a context manager for progress tracking with manual updates.

    This is useful for cases where you need more control over progress updates,
    such as nested progress bars or custom task tracking.

    Args:
        backend (str): Progress bar backend to use. One of ['rich', 'tqdm', 'none', 'simple'].
            Defaults to 'none'.
        **kwargs: Additional keyword arguments to pass to the backend progress bar context.

    Yields:
        Context manager for progress tracking, or nullcontext() if backend is 'none'.

    Example:
        >>> with get_progress_context('rich') as progress:
        ...     task = progress.add_task("Processing", total=100)
        ...     for i in range(100):
        ...         process(i)
        ...         progress.update(task, advance=1)
    """
    # Normalize backend names
    if backend in ("simple", "none", None):
        backend = "none"
    elif backend not in ("rich", "tqdm"):
        logger.warning(f"Unknown progress bar backend '{backend}'. Defaulting to 'none'.")
        backend = "none"

    if backend == "none":
        yield nullcontext()

    elif backend == "rich":
        try:
            from rich.progress import Progress

            with Progress(**kwargs) as progress:
                yield progress
        except ImportError:
            logger.warning(
                "rich is not installed. Install it with 'pip install rich' to use rich progress bars. "
                "Falling back to no progress tracking."
            )
            yield nullcontext()

    elif backend == "tqdm":
        # tqdm doesn't have a good context manager for manual progress tracking
        # We'll just yield a nullcontext and let users use tqdm directly if needed
        logger.warning("Manual progress context is not well-supported with tqdm. Consider using 'rich' or 'none'.")
        yield nullcontext()


def get_progress_bar_callback(backend: str = "none", **kwargs):
    """Get a PyTorch Lightning progress bar callback based on the backend.

    Args:
        backend (str): Progress bar backend to use. One of ['rich', 'tqdm', 'none', 'simple'].
            Defaults to 'none'.
        **kwargs: Additional keyword arguments to pass to the callback.

    Returns:
        Optional callback instance for PyTorch Lightning, or None if backend is 'none'.

    Example:
        >>> callback = get_progress_bar_callback('rich')
        >>> trainer = pl.Trainer(callbacks=[callback] if callback else [])
    """
    # Normalize backend names
    if backend in ("simple", "none", None):
        return None

    if backend == "rich":
        try:
            from pytorch_lightning.callbacks import RichProgressBar

            return RichProgressBar(**kwargs)
        except ImportError:
            logger.warning(
                "rich is not installed or RichProgressBar is not available. "
                "Install it with 'pip install rich' to use rich progress bars in PyTorch Lightning. "
                "Falling back to default progress bar."
            )
            return None

    elif backend == "tqdm":
        try:
            from pytorch_lightning.callbacks import TQDMProgressBar

            return TQDMProgressBar(**kwargs)
        except ImportError:
            logger.warning(
                "TQDMProgressBar is not available in this version of PyTorch Lightning. "
                "Falling back to default progress bar."
            )
            return None
    else:
        logger.warning(f"Unknown progress bar backend '{backend}'. No callback will be added.")
        return None
