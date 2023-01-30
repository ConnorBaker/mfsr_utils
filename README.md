# mfsr_utils

Utility functions and data for Multi-frame super-resolution.

Contains pipelines for producing RAW bursts from a single RAW image.

Also contains a number of datasets for training and evaluating MFSR models.

Install with

```bash
pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu117 .[lint,typecheck,test]
```

Thank you to the authors involved in the following work, from which this code is derived:

- <http://timothybrooks.com/tech/unprocessing>
- <https://github.com/Algolzw/BSRT>

## TODO

- [ ] Move to DataPipes from the current Dataset implementation
    - Tracked in [pipelines: Research `torchvision` offerings](https://github.com/ConnorBaker/mfsr_utils/issues/3)
- [ ] Refactor to remove OpenCV usage from `single2lrburst`
    - Tracked in [single2lrburst: Refactor to remove OpenCV usage](https://github.com/ConnorBaker/mfsr_utils/issues/10)
- [x] Refactor to remove dependency on PyTorch Lightning
