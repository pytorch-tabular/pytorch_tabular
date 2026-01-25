# History

## 1.2.0 (2026-01-26)

* Compatibility with Python up to version 3.14, end-of-life Python 3.8 and 3.9


### Enhancements

* [ENH] Protection for MDN Head misuse by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/448
* [ENH] Remove restriction for using missing and unknown category in SSL models by @sorenmacbeth in https://github.com/pytorch-tabular/pytorch_tabular/pull/470
* [ENH] Add multi target classification by @YonyBresler in https://github.com/pytorch-tabular/pytorch_tabular/pull/441
* [ENH] Feature/tuner multiple model by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/461
* [ENH] Add dataloader_kwargs support in DataConfig by @snehilchatterjee in https://github.com/pytorch-tabular/pytorch_tabular/pull/492
* [ENH] Adding informative str and repr  by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/507
* [ENH] Fix SSL finetuning bug by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/510
* [ENH] Enable Support for Multi-GPU Training by @sorenmacbeth in https://github.com/pytorch-tabular/pytorch_tabular/pull/517
* [ENH] Add Built-in Support for Model Stacking by @taimo3810 in https://github.com/pytorch-tabular/pytorch_tabular/pull/520
* [ENH] Make tensor dtypes `np.float32` for MPS devices by @sorenmacbeth in https://github.com/pytorch-tabular/pytorch_tabular/pull/540
* [ENH] Optimizer lr scheduler interval by @sorenmacbeth in https://github.com/pytorch-tabular/pytorch_tabular/pull/545
* [ENH] add conditional test skips to estimator specific tests by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/607

### Documentation

* [DOC] Create FUNDING.yml by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/376
* [DOC] Fix syntax error in experiment_tracking.md log_target by @furyhawk in https://github.com/pytorch-tabular/pytorch_tabular/pull/382
* [DOC] Fix documentation structure and change default variable by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/394
* [DOC] docs(contributor): contributors readme action update by @github-actions[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/390
* [DOC] Update README.md by @HernandoR in https://github.com/pytorch-tabular/pytorch_tabular/pull/410
* [DOC] lint: simplify used tools by @Borda in https://github.com/pytorch-tabular/pytorch_tabular/pull/431
* [DOC] Fixed typo in metrics_prob_input by @abhisharsinha in https://github.com/pytorch-tabular/pytorch_tabular/pull/455
* [DOC] README - collected important links at top, badges by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/624

### Maintenance

* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/378
* [MNT] build(deps): update torchmetrics requirement from <1.3.0,>=0.10.0 to >=0.10.0,<1.4.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/391
* [MNT] build(deps): bump actions/cache from 3 to 4 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/392
* [MNT] build(deps): update pytorch-lightning requirement from <2.2.0,>=2.0.0 to >=2.0.0,<2.3.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/409
* [MNT] build(deps): bump pypa/gh-action-pypi-publish from 1.8.11 to 1.8.12 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/407
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/395
* [MNT] build(deps): update plotly requirement from <5.19.0,>=5.13.0 to >=5.13.0,<5.20.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/408
* [MNT] Update mkdocstrings[python] requirement from ==0.22.* to ==0.23.* by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/282
* [MNT] Bump akhilmhdh/contributors-readme-action from 2.3.6 to 2.3.10 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/454
* [MNT] freeze `numpy <2.0` & fix ci+docs by @Borda in https://github.com/pytorch-tabular/pytorch_tabular/pull/482
* [MNT] Update pytorch-lightning requirement from <2.3.0,>=2.0.0 to >=2.0.0,<2.5.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/476
* [MNT] Bump pypa/gh-action-pypi-publish from 1.8.12 to 1.10.1 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/484
* [MNT] Update torchmetrics requirement from <1.4.0,>=0.10.0 to >=0.10.0,<1.5.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/477
* [MNT] Update plotly requirement from <5.20.0,>=5.13.0 to >=5.13.0,<5.25.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/478
* [MNT] Update protobuf requirement from <4.26.0,>=3.20.0 to >=3.20.0,<5.29.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/479
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/417
* [MNT] docs(contributor): contributors readme action update by @github-actions[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/468
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/485
* [MNT] docs(contributor): contributors readme action update by @github-actions[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/488
* [MNT] docs(contributor): contributors readme action update by @github-actions[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/505
* [MNT] Update wandb requirement from <0.17.0,>=0.15.0 to >=0.15.0,<0.19.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/489
* [MNT] Update torchmetrics requirement from <1.5.0,>=0.10.0 to >=0.10.0,<1.6.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/501
* [MNT] Update .pre-commit-config.yaml by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/506
* [MNT] Bump pypa/gh-action-pypi-publish from 1.10.1 to 1.11.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/500
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/486
* [MNT] Update broken link for Denoising AutoEncoder tutorial by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/511
* [MNT] Bump pypa/gh-action-pypi-publish from 1.11.0 to 1.12.2 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/513
* [MNT] Update protobuf requirement from <5.29.0,>=3.20.0 to >=3.20.0,<5.30.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/516
* [MNT] Update torchmetrics requirement from <1.6.0,>=0.10.0 to >=0.10.0,<1.7.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/514
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/509
* [MNT] Update mkdocstrings[python] requirement from ==0.26.* to ==0.27.* by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/515
* [MNT] chore(ci): setting concurrency by @Borda in https://github.com/pytorch-tabular/pytorch_tabular/pull/524
* [MNT] docs(contributor): contributors readme action update by @github-actions[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/523
* [MNT] Update base.txt by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/556
* [MNT] docs(contributor): contributors readme action update by @github-actions[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/560
* [MNT] Update mkdocstrings[python] requirement from ==0.27.* to ==0.29.* by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/559
* [MNT] Migrate packaging to `pyproject.toml` by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/594
* [MNT] fix CI jobs with problems by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/597
* [MNT] improved `testing` CI job - `uv` and installed dependencies display by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/599
* [MNT] Update dependency versions and complete migration to `pyproject.toml` by @phoeenniixx in https://github.com/pytorch-tabular/pytorch_tabular/pull/596
* [MNT] Bump `actions/checkout` from 4 to 6 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/591
* [MNT] CI test matrix with unix, windows, and all supported python versions by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/604
* [MNT] stop CI job that adds documentation build link to PR description by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/608
* [MNT] remove unused dependencies from `pyproject.toml` by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/602
* [MNT] update `dependabot.yml` for daily updates by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/609
* [MNT] isolate `captum` soft dependency in tests by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/613
* [MNT] [Dependabot](deps): Bump actions/cache from 4 to 5 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/614
* [MNT] add tests without and with all extra dependencies by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/612
* [MNT] remove `requirements` `txt` files after `pyproject.toml` migration by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/620
* [MNT] [Dependabot](deps): Update einops requirement from <0.8.0,>=0.6.0 to >=0.6.0,<0.9.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/618
* [MNT] [Dependabot](deps): Update torchmetrics requirement from <1.7.0,>=0.10.0 to >=0.10.0,<1.9.0 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/619
* [MNT] [Dependabot](deps-dev): Update mkdocs-material requirement from ==9.5.* to >=9.5,<9.8 by @dependabot[bot] in https://github.com/pytorch-tabular/pytorch_tabular/pull/617
* [MNT] isolate `pytorch-tabnet` as soft dependency by @fkiraly in https://github.com/pytorch-tabular/pytorch_tabular/pull/610

### Fixes

* [BUG] KeyError nn.activation in Tab Transformer and FT Transformer by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/385
* [BUG] fix attr that was not been changed by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/384
* [BUG] cannot pickle generator object error by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/383
* [BUG] bug fix for cross validate by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/414
* [BUG] Fix to Tuner change trainer and optimizer configs by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/387
* [BUG] TabularDataModule: Fix pandas returning a series when calling nunique() by @charitarthchugh in https://github.com/pytorch-tabular/pytorch_tabular/pull/420
* [BUG] Bug fix for saving and loading custom loss functions by @manujosephv in https://github.com/pytorch-tabular/pytorch_tabular/pull/415
* [BUG] Fix to Tuner change trainer and optimizer configs >>Change batch_size in tuner by @ProgramadorArtificial in https://github.com/pytorch-tabular/pytorch_tabular/pull/449
* [BUG] Bug fix for "Categorical" dtype by @snehilchatterjee in https://github.com/pytorch-tabular/pytorch_tabular/pull/493
* [BUG] Torch load issue fix with pytorch 2.6 by @ArozHada in https://github.com/pytorch-tabular/pytorch_tabular/pull/543
* [BUG] torch.load fix for all pytorch versions. by @ArozHada in https://github.com/pytorch-tabular/pytorch_tabular/pull/554
* [BUG] minor bug fix by @enifeder in https://github.com/pytorch-tabular/pytorch_tabular/pull/551
* [BUG] change progress bar default to `simple` and decouple `rich` core dependency to avoid default failure by @fkiraly and @phoeenniixx in https://github.com/pytorch-tabular/pytorch_tabular/pull/601

## New Contributors
* @furyhawk made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/382
* @HernandoR made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/410
* @charitarthchugh made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/420
* @abhisharsinha made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/455
* @YonyBresler made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/441
* @snehilchatterjee made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/492
* @taimo3810 made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/520
* @ArozHada made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/543
* @enifeder made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/551
* @fkiraly made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/594
* @phoeenniixx made their first contribution in https://github.com/pytorch-tabular/pytorch_tabular/pull/596


---
## 1.1.0 (2024-01-15)

### New Features and Enhancements
- **Added DANet Model**: Added a new model, DANet, for tabular data.
- **Explainability**: Integrated Captum for explainability
- **Hyperparameter Tuner:** Added Grid and Random Search functionality to search through hyperparameters and return best model.
- **Model Sweep:** Added an easy "Model Sweep" method with which we can sweep a list of models with given data and quickly assess performance.
- **Documentation Enhancements:** Improved documentation to make it more user-friendly and informative
- **Dependency Updates:** Updated various dependencies for improved compatibility and security
- **Graceful Out-of-Memory Handling:** Added graceful out-of-memory handling for tabular models
- **GhostBatchNorm:** Added GhostBatchNorm to the library

### Deprecations
- **Deprecations:** Handled deprecations and updated the library accordingly
- **Entmax Dependency Removed:** Removed dependency on entmax

### Infrastructure and CI/CD
- **Continuous Integration:** Improved CI with new actions and labels
- **Dependency Management:** Updated dependencies and restructured requirements

### API Changes
- [BREAKING CHANGE] **SSL API Change:** Addressed SSL API change, along with documentation and tutorial updates.
- **Model Changes:** Added is_fitted and other markers to the tabular model.
- **Custom Optimizer:** Allow custom optimizer in the model config.

### Contributors
- Thanks to all the contributors who helped shape this release! ([List of Contributors](Link_to_Contributors))

### Upgrading
- Ensure to check the updated documentation for any breaking changes or new features.
- If you are using SSL, please check the updated API and documentation.

## 1.0.2 (2023-05-31)

### New Features:

- Added Feature Importance: The library now includes a new method in TabularModel and BaseModel for enabling feature importance. Feature Importance has been enabled for FTTransformer and GATE models. [Commit: dc2a49e]
### Enhancements:

- Enabled two more parameters in the GATE model. [Commit: 3680413]
- Included metric_prob_input parameter in the library configuration. This update allows for better control over metrics in the models. [Commit: 0612db5]
- Slight improvements to the GATE model, including changes to defaults for better performance. [Commit: c30a6c3]
- Minor bug fixes and improvements, including accelerator options in the configuration and progress bar enhancements. [Commit: f932230, bdd9adb, f932230]
### Dependency Updates:

- Updated dependencies, including docformatter, pyupgrade, and ruff-pre-commit. [Commits: 4aae9a8, b3df4ce, bdd9adb, 55e800c, c6c4679, c01154b, 107cd2f]
### Documentation Updates:

- Updated the library's README.md file. [Commits: db8f3b2, cab6bf1, 669faec, 1e6c400, 3097799, 7fabf6b]
### Other Improvements:

- Various code optimizations, bug fixes, and CI enhancements. [Commits: 5637020, e5171bf, 812b40f]

For more details, you can refer to the respective commits on the library's GitHub repository.

## 1.0.1 (2023-01-20)

- Bugfix for default metric for binary classification




## 1.0.0 (2023-01-18)

- Added a new task - Self Supervised Learning (SSL) and a separate training API for it.
- Added new SOTA model - Gated Additive Tree Ensembles (GATE).
- Added one SSL model - Denoising AutoEncoder.
- Added lots of new tutorials and updated entire documentation.
- Improved code documentation and type hints.
- Separated a Model into separate Embedding, Backbone, and Head.
- Refactored all models to separate Backbone as native PyTorch Model(nn.Module).
- Refactored commonly used modules (layers, activations etc. to a common module).
- Changed MixedDensityNetworks completely (breaking change). Now MDN is a head you can use with any model.
- Enabled a low level api for training model.
- Enabled saving and loading of datamodule.
- Added trainer_kwargs to pass any trainer argument PyTorch Lightning supports.
- Added Early Stopping and Model Checkpoint kwargs to use all the arguments in PyTorch Lightining.
- Enabled prediction using GPUs in predict method.
- Added `reset_model` to reset model weights to random.
- Added many save and load functions including ONNX(experimental).
- Added random seed as a parameter.
- Switched over completely to Rich progressbars from tqdm.
- Fixed class-balancing / mu propagation and set default to 1.0.
- Added PyTorch Profiler for debugging performance issues.
- Fixed bugs with FTTransformer and TabTransformer.
- Updated MixedDensityNetworks fixing a bug with lambda_pi.
- Many CI/CD improvements including complete integration with GitHub Actions.
- Upgraded all dependencies, including PyTorch Lightning, pandas, to latest versions and added dependabot to manage it going forward.
- Added pre-commit to ensure code integrity and standardization.

## 0.7.0 (2021-09-01)

- Implemented TabTransformer and FTTransformer models
- Included capability to save a model using GPU an load in CPU
- Made the temp folder pytorch tabular specific to avoid conflicts with other tmp folders.
- Some bug fixes
- Edited an error out of Advanced Tutorial in docs

## 0.6.0 (2021-06-21)

- Upgraded versions of PyTorch Lightning to 1.3.6
- Changed the way `gpus` parameter is handled to avoid confusion. `None` is CPU, `-1` is all GPUs, `int` is number of GPUs
- Added a few more Trainer Params like `deterministic`, `auto_select_gpus`
- Some bug fixes and changes to docs
- Added `seed_everything` to the fit method to ensure reproducibility
- Refactored data_aware_initialization to be part of the BaseModel. Inherited Models can override the method to implement data aware initialization techniques

## 0.5.0 (2021-03-18)

- Added more documentation
- Added Zenodo citation

## 0.4.0 (2021-03-18)

- Added AutoInt Model
- Added Mixture Density Networks
- Refactored the classes to separate backbones from the head of the models
- Changed the saving and loading model to work for custom parameters that you pass in `fit`

## 0.3.0 (2021-03-02)

- Fixed a bug on inference

## 0.2.0 (2021-02-07)

- Fixed an issue with torch.clip and torch version
- Fixed an issue with `gpus` parameter in TrainerConfig, by setting default value to `None` for CPU
- Added feature to use custom sampler in the training dataloader
- Updated documentation and added a new tutorial for imbalanced classification

## 0.0.1 (2021-01-26)

- First release on PyPI.
