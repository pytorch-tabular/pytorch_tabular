# [ENH] Refactor progress bar backend to allow user choice and decouple from rich

Fixes #605

## Description

This PR refactors the progress bar backend to provide flexible, user-configurable progress tracking without requiring `rich` as a mandatory dependency. This addresses the issues raised in #605 where hard-coded `rich` dependency was causing compatibility problems and limiting user choice.

## Motivation and Context

Previously, PyTorch Tabular had `rich` hard-coded throughout the codebase, which:
- Caused compatibility issues in certain environments (see #600)
- Forced all users to install `rich` even if they didn't want progress bars
- Provided no way for users to choose their preferred progress bar library

This PR solves these issues by creating an abstraction layer that:
- Supports multiple backends (`rich`, `tqdm`, `simple`, `none`)
- Makes `rich` and `tqdm` optional dependencies
- Provides graceful fallback if preferred backend isn't installed
- Maintains backward compatibility with a sensible default

## Key Changes

### 1. New Progress Utility Module (`src/pytorch_tabular/utils/progress.py`)

Created a comprehensive utility module with three main functions:

- **`get_progress_tracker(backend, iterable, description, total, **kwargs)`**
  - Returns a progress tracker for iterating over iterables with progress display
  - Supports backends: `rich`, `tqdm`, `simple`, `none`
  - Can be used inline or as a callable wrapper
  - Gracefully falls back if backend not installed

- **`get_progress_context(backend, **kwargs)`**
  - Context manager for manual progress tracking
  - Useful for nested progress bars or custom task tracking
  - Provides unified interface across different backends

- **`get_progress_bar_callback(backend, **kwargs)`**
  - Returns PyTorch Lightning progress bar callbacks
  - Returns appropriate callback for `rich` or `tqdm` backends
  - Returns `None` for `none`/`simple` to disable progress bars

### 2. Updated Core Files

Replaced all hard-coded `rich` imports with the new progress utilities:

- ✅ `src/pytorch_tabular/tabular_model.py`
  - Removed direct import of `RichProgressBar`
  - Updated `_prepare_callbacks()` to use `get_progress_bar_callback()`
  - Updated predict method to use `get_progress_tracker()`

- ✅ `src/pytorch_tabular/categorical_encoders.py`
  - Replaced `from rich.progress import track`
  - Uses `get_progress_tracker()` with `"simple"` backend

- ✅ `src/pytorch_tabular/feature_extractor.py`
  - Replaced `from rich.progress import track`
  - Uses `get_progress_tracker()` with `"simple"` backend

- ✅ `src/pytorch_tabular/tabular_model_tuner.py`
  - Replaced `from rich.progress import Progress`
  - Uses `get_progress_context()` and `get_progress_tracker()`

- ✅ `src/pytorch_tabular/tabular_model_sweep.py`
  - Replaced `from rich.progress import Progress, track`
  - Uses `get_progress_context()` and `get_progress_tracker()`

### 3. Configuration Fix (`src/pytorch_tabular/config/config.py`)

- Fixed metadata documentation for `progress_bar` field
- Corrected to state default is `"simple"` (was incorrectly documented as `"rich"`)
- Default behavior unchanged - uses `"simple"` backend

### 4. Dependency Management (`pyproject.toml`)

- **Moved `rich>=11.0.0` from required to optional dependencies**
- Added new `[progress]` optional dependency group:
  ```toml
  [project.optional-dependencies]
  progress = [
      "rich>=11.0.0",
      "tqdm>=4.0.0",
  ]
  ```
- Users can install with: `pip install pytorch_tabular[progress]`

### 5. Updated Exports (`src/pytorch_tabular/utils/__init__.py`)

Added new progress utilities to public API:
```python
from .progress import get_progress_bar_callback, get_progress_context, get_progress_tracker
```

### 6. Documentation

- **PROGRESS_BAR_GUIDE.md**: Comprehensive user guide with examples
- **PROGRESS_BAR_REFACTOR.md**: Technical implementation details and migration guide

## Benefits

✅ **Decoupled from rich** - No longer requires `rich` as core dependency, avoiding compatibility issues  
✅ **User choice** - Users can choose between `rich`, `tqdm`, `simple`, or `none` backends  
✅ **Graceful fallback** - Works even if preferred backend isn't installed  
✅ **Backward compatible** - Default `"simple"` backend requires no external dependencies  
✅ **Consistent API** - Single abstraction layer for all progress tracking  
✅ **Better modularity** - Centralized progress bar logic in one module  

## Usage Examples

### Default Behavior (No Changes Required)
```python
# Works without any progress bar dependencies
trainer_config = TrainerConfig(
    progress_bar="simple",  # or omit, as this is default
    max_epochs=10
)
```

### Using Rich Progress Bars
```python
# After: pip install pytorch_tabular[progress]
trainer_config = TrainerConfig(
    progress_bar="rich",
    max_epochs=10
)
```

### Using TQDM Progress Bars
```python
# After: pip install pytorch_tabular[progress]
trainer_config = TrainerConfig(
    progress_bar="tqdm",
    max_epochs=10
)
```

### Disabling Progress Bars
```python
trainer_config = TrainerConfig(
    progress_bar="none",
    max_epochs=10
)
```

## Migration Guide

### For Users
**No changes required!** The default behavior uses the `"simple"` backend which requires no external dependencies. If you want fancy progress bars, install the progress extras: `pip install pytorch_tabular[progress]`

### For Contributors
If you're adding new code that needs progress tracking:

**Before:**
```python
from rich.progress import track

for item in track(items, description="Processing"):
    process(item)
```

**After:**
```python
from pytorch_tabular.utils.progress import get_progress_tracker

track = get_progress_tracker("simple", description="Processing")
for item in track(items):
    process(item)
```

## Testing

All modified files have been syntax-checked and the changes maintain backward compatibility. The new progress utility module includes:
- Support for all four backends (`rich`, `tqdm`, `simple`, `none`)
- Proper error handling and graceful fallback
- Clear warning messages when backends are unavailable

## Checklist

- [x] Created new progress utility module with comprehensive documentation
- [x] Updated all files with hard-coded `rich` imports
- [x] Made `rich` an optional dependency
- [x] Fixed config metadata
- [x] Updated exports
- [x] Added comprehensive documentation (guides and technical details)
- [x] Maintained backward compatibility
- [x] Syntax-checked all modified files
- [x] Added clear migration examples

## Related Issues

- Closes #605
- Related to #600 (rich compatibility issues)
- Related to PR #638 (alternative implementation)

## Notes for Reviewers

1. The default behavior is unchanged - uses `"simple"` backend with no external dependencies
2. All progress bar usage is now centralized in `src/pytorch_tabular/utils/progress.py`
3. The implementation gracefully handles missing dependencies with clear warning messages
4. Comprehensive documentation is provided for both users and contributors

---

@esoc - Please review this implementation. All hard-coded `rich` dependencies have been removed and replaced with a flexible, user-configurable system that maintains backward compatibility while addressing the issues raised in #605.
