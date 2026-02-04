# Issue #611: Dealing with pytorch-tabnet Lapsed Dependency

## Problem Statement

The `pytorch-tabnet` dependency, which is used for TabNet model implementation in PyTorch Tabular, has not been actively maintained since July 2023 (last release v4.1.0). This poses several risks:

1. **Security vulnerabilities**: No updates to address potential security issues
2. **Compatibility issues**: May break with newer versions of PyTorch, Python, or other dependencies  
3. **Bug fixes**: No ongoing bug fixes or improvements
4. **Feature stagnation**: Cannot benefit from community improvements

**Current status in pytorch_tabular:**
- `pytorch-tabnet<4.2` is in optional `extra` dependencies
- Used only for TabNet model via soft dependency checks
- Already has graceful fallback when not installed

## Recommended Solution: Vendor and Maintain Internally

Given that `pytorch-tabnet` is already an **optional dependency** and TabNet is **one specific model** among many in PyTorch Tabular, I recommend **vendoring** (copying and maintaining internally) the TabNet implementation.

### Why Vendoring is the Best Option:

1. ✅ **Control**: Full control over maintenance, bug fixes, and updates
2. ✅ **Stability**: No external dependency breaking changes
3. ✅ **Security**: Can patch security issues immediately
4. ✅ **Compatibility**: Can ensure compatibility with PyTorch Tabular's ecosystem
5. ✅ **Simplicity**: Reduces external dependencies
6. ✅ **License Compatible**: pytorch-tabnet uses MIT license (same as pytorch_tabular)

### Why Not Other Options:

**Forking to separate repo:**
- ❌ Still external dependency
- ❌ More maintenance overhead
- ❌ Users still need to install separately

**Contacting original maintainers:**
- ❌ 3 years of inactivity suggests low chance of success
- ❌ Even if successful, still external dependency
- ❌ No guarantee of long-term maintenance

## Implementation Plan

### Phase 1: Vendor the Code (Immediate)

1. **Copy TabNet implementation into pytorch_tabular:**
   ```
   src/pytorch_tabular/models/tabnet/
   ├── __init__.py
   ├── tabnet_model.py (existing - wrapper)
   ├── tabnet_config.py (existing - config)  
   ├── _vendored/  (NEW)
   │   ├── __init__.py
   │   ├── tab_network.py  (from pytorch_tabnet.tab_network)
   │   ├── abstract_model.py  (from pytorch_tabnet.abstract_model)
   │   ├── sparsemax.py  (from pytorch_tabnet.sparsemax)
   │   ├── utils.py  (from pytorch_tabnet.utils)
   │   └── LICENSE  (MIT license from pytorch-tabnet)
   ```

2. **Update imports in tabnet_model.py:**
   - Change from `pytorch_tabnet.tab_network` to internal `_vendored.tab_network`
   - Remove soft dependency checks for `pytorch-tabnet`

3. **Add clear attribution:**
   - Include original LICENSE file
   - Add comments crediting dreamquark-ai/tabnet
   - Document in README and docs

### Phase 2: Update Dependencies (Immediate)

1. **Remove from optional dependencies:**
   ```toml
   # In pyproject.toml - REMOVE:
   # "pytorch-tabnet<4.2",
   ```

2. **Update documentation:**
   - Remove references to installing pytorch-tabnet
   - Update TabNet docs to mention vendored implementation

3. **Update tests:**
   - Remove `_check_soft_dependencies("pytorch-tabnet")` checks
   - Tests should now always run for TabNet

### Phase 3: Modernize and Maintain (Ongoing)

1. **Code improvements:**
   - Update for latest PyTorch compatibility
   - Fix any linting issues
   - Add type hints
   - Improve documentation

2. **Testing:**
   - Ensure comprehensive test coverage
   - Add regression tests
   - Test with multiple PyTorch versions

3. **Features:**
   - Consider community improvements from forks
   - Evaluate and integrate useful features

## Files to Vendor from pytorch-tabnet

Based on pytorch_tabular's current usage, we need these core files:

### Essential Files:
1. **`tab_network.py`** - Core TabNet architecture
2. **`abstract_model.py`** - Base model class  
3. **`sparsemax.py`** - Sparsemax activation
4. **`utils.py`** - Utility functions (especially `create_group_matrix`)

### Supporting Files:
5. **`multiclass_utils.py`** - Multi-class classification utilities
6. **`metrics.py`** - Evaluation metrics

### Not Needed:
- `tab_model.py` - TabNet sklearn wrapper (pytorch_tabular has its own)
- `pretraining.py` - Pretraining logic (if not used)
- `callbacks.py` - Custom callbacks (pytorch_tabular uses PyTorch Lightning)
- `augmentations.py` - Data augmentation (if not used)

## License Compliance

✅ **License Compatible:**
- pytorch-tabnet: MIT License
- pytorch_tabular: MIT License
- Can vendor with proper attribution

**Required Attribution:**
```python
# File header for vendored files:
# Vendored from pytorch-tabnet (https://github.com/dreamquark-ai/tabnet)
# Original work Copyright (c) 2019-2023 DreamQuark
# Licensed under MIT License (see _vendored/LICENSE)
# Modified for integration with PyTorch Tabular
```

## Migration Path for Users

### Before (Current):
```bash
pip install pytorch_tabular[extra]  # includes pytorch-tabnet
```

### After (Vendored):
```bash
pip install pytorch_tabular  # TabNet included by default
```

**Backwards Compatibility:**
- Users don't need to change any code
- TabNet will work out of the box
- No breaking changes to API

## Alternative: Conditional Vendoring

If you want to maintain some backwards compatibility:

```python
# In tabnet_model.py
try:
    # Try to import from installed pytorch-tabnet first
    from pytorch_tabnet.tab_network import TabNet
    from pytorch_tabnet.utils import create_group_matrix
    _USING_VENDORED = False
except ImportError:
    # Fall back to vendored version
    from ._vendored.tab_network import TabNet
    from ._vendored.utils import create_group_matrix
    _USING_VENDORED = True
    logger.info("Using vendored TabNet implementation")
```

This allows users who have pytorch-tabnet installed to continue using it, while new users automatically get the vendored version.

## Risks and Mitigation

### Risks:
1. **Maintenance burden**: Need to maintain TabNet code
2. **Bug discovery**: May find bugs in vendored code
3. **Community resistance**: Some may prefer external dependency

### Mitigation:
1. **Maintenance**: TabNet is stable; minimal maintenance needed
2. **Bugs**: Can fix immediately (vs waiting for upstream)
3. **Community**: Explain benefits and provide migration guide

## Estimated Effort

- **Phase 1 (Vendoring)**: 4-6 hours
  - Copy files
  - Update imports
  - Add attribution
  - Basic testing

- **Phase 2 (Cleanup)**: 2-3 hours
  - Update dependencies
  - Update documentation
  - Update tests

- **Phase 3 (Ongoing)**: As needed
  - Monitor for issues
  - Apply fixes/improvements
  - Respond to community feedback

**Total initial effort**: ~8-10 hours

## Recommendation Summary

**I recommend vendoring the pytorch-tabnet implementation** for the following reasons:

1. ✅ TabNet is already optional - low risk
2. ✅ MIT license allows vendoring with attribution
3. ✅ Gives full control over maintenance
4. ✅ Eliminates external dependency issues
5. ✅ Can modernize and improve code
6. ✅ Seamless user experience (no extra install needed)

This approach provides the best balance of:
- **Control**: Full ownership and maintenance ability
- **Stability**: No external breaking changes
- **User Experience**: Works out of the box
- **Maintainability**: Clear scope, stable codebase
- **License Compliance**: Proper attribution

## Next Steps

1. ✅ Get approval from pytorch_tabular maintainers
2. ⬜ Create a new branch for vendoring work
3. ⬜ Copy necessary files from pytorch-tabnet
4. ⬜ Update imports and dependencies
5. ⬜ Add comprehensive tests
6. ⬜ Update documentation
7. ⬜ Create PR with changes
8. ⬜ Announce in release notes

## Questions for Maintainers

Before implementing, please confirm:

1. Do you agree with the vendoring approach?
2. Any specific features from pytorch-tabnet we need to preserve?
3. Should we maintain backwards compatibility with installed pytorch-tabnet?
4. Any naming preferences for the vendored module?
5. Timeline expectations for implementation?

---

**References:**
- Original Issue: #611
- pytorch-tabnet repo: https://github.com/dreamquark-ai/tabnet
- Last release: v4.1.0 (July 23, 2023)
- License: MIT
