# Progress Bar Configuration Guide

## Overview

PyTorch Tabular now supports flexible progress bar backends, allowing you to choose between different progress tracking libraries or disable progress bars entirely. This guide explains how to configure and use different progress bar backends.

## Supported Backends

### 1. `"simple"` (Default)
- **Description**: Basic progress tracking without requiring external dependencies
- **Dependencies**: None (built into Python)
- **Use Case**: Default option when you want minimal dependencies
- **Example**:
```python
from pytorch_tabular import TabularModel
from pytorch_tabular.config import TrainerConfig

trainer_config = TrainerConfig(
    progress_bar="simple",  # or omit entirely as this is the default
    max_epochs=10
)
```

### 2. `"rich"`
- **Description**: Beautiful, feature-rich progress bars with advanced formatting
- **Dependencies**: `pip install rich` or `pip install pytorch_tabular[progress]`
- **Use Case**: When you want visually appealing progress tracking with colors and formatting
- **Example**:
```python
trainer_config = TrainerConfig(
    progress_bar="rich",
    max_epochs=10
)
```

### 3. `"tqdm"`
- **Description**: Popular, lightweight progress bars
- **Dependencies**: `pip install tqdm` or `pip install pytorch_tabular[progress]`
- **Use Case**: When you prefer tqdm's style or have it already installed
- **Example**:
```python
trainer_config = TrainerConfig(
    progress_bar="tqdm",
    max_epochs=10
)
```

### 4. `"none"`
- **Description**: Disables all progress bars
- **Dependencies**: None
- **Use Case**: When running in production, CI/CD, or when you don't want any progress output
- **Example**:
```python
trainer_config = TrainerConfig(
    progress_bar="none",
    max_epochs=10
)
```

## Installation

### Minimal Installation (No Progress Bars)
```bash
pip install pytorch_tabular
# Uses "simple" backend by default (no external dependencies)
```

### With Progress Bar Support
```bash
# Install with both rich and tqdm support
pip install pytorch_tabular[progress]

# Or install specific backends
pip install pytorch_tabular rich      # for rich support
pip install pytorch_tabular tqdm      # for tqdm support
```

## Usage Examples

### Basic Training with Progress Bar
```python
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    TrainerConfig,
    OptimizerConfig,
    ExperimentConfig
)
from pytorch_tabular.models import CategoryEmbeddingModelConfig

# Configure with your preferred progress bar
trainer_config = TrainerConfig(
    batch_size=64,
    max_epochs=10,
    progress_bar="rich"  # or "tqdm", "simple", "none"
)

# Create and train model
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tabular_model.fit(train=train_df, validation=val_df)
```

### Prediction with Progress Tracking
```python
# The progress bar backend is automatically used during prediction
predictions = tabular_model.predict(
    test_df,
    progress_bar="rich"  # Can specify different backend for predictions
)
```

### Model Sweep with Progress Tracking
```python
from pytorch_tabular import model_sweep

results = model_sweep(
    task="classification",
    train=train_df,
    test=test_df,
    data_config=data_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    progress_bar=True,  # Uses the trainer_config progress_bar setting
    verbose=True
)
```

## Graceful Fallback

If you specify a backend that isn't installed, PyTorch Tabular will:
1. Log a warning message
2. Fall back to no progress bar
3. Continue execution normally

Example:
```python
# If rich is not installed but you specify it
trainer_config = TrainerConfig(progress_bar="rich")

# You'll see a warning like:
# "rich is not installed. Install it with 'pip install rich' to use rich progress bars. 
#  Falling back to no progress bar."

# But training will continue normally without errors
```

## Advanced: Using Progress Utilities Directly

For advanced users who want to use the progress utilities in custom code:

```python
from pytorch_tabular.utils.progress import (
    get_progress_tracker,
    get_progress_context,
    get_progress_bar_callback
)

# Simple iteration with progress
track = get_progress_tracker("rich", description="Processing items")
for item in track(my_items):
    process(item)

# Manual progress updates
with get_progress_context("rich") as progress:
    task = progress.add_task("Processing", total=100)
    for i in range(100):
        do_work(i)
        progress.update(task, advance=1)

# Get PyTorch Lightning callback
callback = get_progress_bar_callback("tqdm")
trainer = pl.Trainer(callbacks=[callback] if callback else [])
```

## Migration from Old Code

### Before (Hard-coded Rich)
```python
from rich.progress import track

for item in track(items, description="Processing"):
    process(item)
```

### After (Configurable Backend)
```python
from pytorch_tabular.utils.progress import get_progress_tracker

track = get_progress_tracker("rich", description="Processing")
for item in track(items):
    process(item)
```

## Troubleshooting

### Progress Bar Not Showing
1. Check that `progress_bar` is not set to `"none"`
2. Verify the backend is installed: `pip install rich` or `pip install tqdm`
3. Ensure `enable_progress_bar=True` in `trainer_kwargs` (default)

### ImportError for Rich/TQDM
Install the required package:
```bash
pip install pytorch_tabular[progress]
# or
pip install rich tqdm
```

### Progress Bar Conflicts
If you're using a specific backend and experiencing issues:
1. Update to the latest version: `pip install --upgrade rich` or `pip install --upgrade tqdm`
2. Try a different backend: switch from `"rich"` to `"tqdm"` or `"simple"`
3. Disable progress bars: set `progress_bar="none"`

## Performance Considerations

- **simple**: Minimal overhead, fastest option
- **rich**: Slightly more overhead due to formatting, but still very fast
- **tqdm**: Similar to rich in performance
- **none**: Zero overhead, best for production deployments

## Best Practices

1. **Development**: Use `"rich"` or `"tqdm"` for nice visual feedback
2. **Production**: Use `"none"` or `"simple"` to reduce overhead
3. **CI/CD**: Use `"none"` to avoid cluttering logs
4. **Jupyter Notebooks**: `"rich"` or `"tqdm"` work great
5. **Debugging**: Use `"none"` to reduce clutter in debug logs

## Future Enhancements

Planned improvements:
- Custom progress bar implementations
- More fine-grained control over progress display
- Integration with other monitoring tools

## Feedback

If you have suggestions for improving progress bar functionality, please open an issue on the GitHub repository.
