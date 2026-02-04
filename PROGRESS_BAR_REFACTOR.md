# Issue #605 - Progress Bar Backend Refactoring

## Summary

This PR addresses issue #605 by refactoring the progress bar backend to allow user choice and decoupling from `rich` as a required dependency.

## Changes Made

### 1. New Progress Utility Module (`src/pytorch_tabular/utils/progress.py`)

Created a comprehensive utility module that abstracts progress bar functionality with three main functions:

- **`get_progress_tracker()`**: Returns a progress tracker for iterating over iterables with progress display
  - Supports backends: `rich`, `tqdm`, `none`, `simple`
  - Falls back gracefully if the requested backend is not installed
  - Can be used inline or as a callable wrapper

- **`get_progress_context()`**: Context manager for manual progress tracking
  - Useful for nested progress bars or custom task tracking
  - Provides a unified interface across different backends

- **`get_progress_bar_callback()`**: Returns PyTorch Lightning progress bar callbacks
  - Returns appropriate callback for `rich` or `tqdm` backends
  - Returns `None` for `none`/`simple` to disable progress bars

### 2. Configuration Update (`src/pytorch_tabular/config/config.py`)

- Fixed metadata documentation for `progress_bar` field to correctly state default is `"simple"` (was incorrectly documented as `"rich"`)
- Default remains `"simple"` which uses basic progress tracking without requiring external dependencies

### 3. Refactored Core Files

All hard-coded `rich` imports have been replaced with the new progress utility:

#### `src/pytorch_tabular/tabular_model.py`
- Removed direct import of `RichProgressBar`
- Updated `_prepare_callbacks()` to use `get_progress_bar_callback()`
- Updated predict method to use `get_progress_tracker()` instead of conditional imports

#### `src/pytorch_tabular/categorical_encoders.py`
- Replaced `from rich.progress import track` with `get_progress_tracker()`
- Uses `"simple"` backend for encoding operations

#### `src/pytorch_tabular/feature_extractor.py`
- Replaced `from rich.progress import track` with `get_progress_tracker()`
- Uses `"simple"` backend for feature generation

#### `src/pytorch_tabular/tabular_model_tuner.py`
- Replaced `from rich.progress import Progress` with progress utilities
- Updated to use `get_progress_context()` and `get_progress_tracker()`
- Maintains backward compatibility with progress bar behavior

#### `src/pytorch_tabular/tabular_model_sweep.py`
- Replaced `from rich.progress import Progress, track` with progress utilities
- Updated to use `get_progress_context()` and `get_progress_tracker()`
- Progress backend defaults to `"simple"` when enabled

### 4. Dependency Management (`pyproject.toml`)

- **Moved `rich>=11.0.0` from required dependencies to optional dependencies**
- Added new `[progress]` optional dependency group containing:
  - `rich>=11.0.0`
  - `tqdm>=4.0.0`
- Users can now install with progress bar support using:
  ```bash
  pip install pytorch_tabular[progress]
  ```

### 5. Exports (`src/pytorch_tabular/utils/__init__.py`)

Added the new progress utilities to the public API:
- `get_progress_tracker`
- `get_progress_context`
- `get_progress_bar_callback`

## Benefits

1. **Decoupled from `rich`**: The package no longer requires `rich` as a core dependency, avoiding compatibility issues
2. **User Choice**: Users can choose their preferred progress bar backend (`rich`, `tqdm`, or none)
3. **Graceful Fallback**: If a requested backend is not installed, the code gracefully falls back to no progress bar with a warning
4. **Backward Compatible**: Existing code continues to work with the default `"simple"` backend
5. **Consistent Interface**: All progress bar usage throughout the codebase now uses the same abstraction layer
6. **Better Modularity**: Progress bar functionality is centralized in one module, making it easier to maintain and extend

## Migration Guide for Users

### Current Behavior (Unchanged)
```python
# Default behavior - uses simple progress tracking (no external dependencies)
trainer_config = TrainerConfig(progress_bar="simple")  # or omit, as this is default
```

### Using Rich Progress Bars
```bash
pip install pytorch_tabular[progress]  # or: pip install rich
```
```python
trainer_config = TrainerConfig(progress_bar="rich")
```

### Using TQDM Progress Bars
```bash
pip install pytorch_tabular[progress]  # or: pip install tqdm
```
```python
trainer_config = TrainerConfig(progress_bar="tqdm")
```

### Disabling Progress Bars
```python
trainer_config = TrainerConfig(progress_bar="none")
```

## Testing Recommendations

1. Test with `progress_bar="none"` to ensure no progress bars are shown
2. Test with `progress_bar="simple"` (default) without installing rich/tqdm
3. Test with `progress_bar="rich"` after installing rich
4. Test with `progress_bar="tqdm"` after installing tqdm
5. Test graceful fallback when backend not installed

## Related Issues

- Closes #605
- Related to #600 (rich compatibility issues)
- Related to PR #638 (alternative implementation)
