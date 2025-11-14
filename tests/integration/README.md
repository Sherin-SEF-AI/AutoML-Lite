# NAS Integration Tests

This directory contains integration tests for the Neural Architecture Search (NAS) feature.

## Test Files

### test_nas_end_to_end.py
Tests complete end-to-end NAS workflows:
- Complete search on small datasets with different strategies (evolutionary, RL)
- Checkpoint save and resume functionality
- AutoMLite integration with NAS enabled/disabled
- Search robustness and error handling
- Search statistics tracking

### test_nas_hardware_aware.py
Tests hardware-aware NAS functionality:
- Hardware constraint satisfaction (latency, memory, model size)
- Latency prediction accuracy and correlation with actual measurements
- Relative latency ordering (larger models are slower)
- Different hardware target predictions (CPU, GPU, mobile)
- Constraint checker integration

### test_nas_transfer_learning.py
Tests transfer learning capabilities:
- Architecture reuse reduces search time
- Warm start with similar architectures
- Similarity-based architecture retrieval
- Architecture adaptation to new problems
- Repository integration with search process

## Running the Tests

### Prerequisites
Install required dependencies:
```bash
pip install -e ".[dev]"
```

### Run All Integration Tests
```bash
pytest tests/integration/ -v
```

### Run Specific Test File
```bash
pytest tests/integration/test_nas_end_to_end.py -v
pytest tests/integration/test_nas_hardware_aware.py -v
pytest tests/integration/test_nas_transfer_learning.py -v
```

### Run Specific Test Class
```bash
pytest tests/integration/test_nas_end_to_end.py::TestEndToEndNAS -v
pytest tests/integration/test_nas_hardware_aware.py::TestHardwareConstraintSatisfaction -v
pytest tests/integration/test_nas_transfer_learning.py::TestArchitectureReuse -v
```

### Run with Coverage
```bash
pytest tests/integration/ --cov=automl_lite.nas --cov-report=html
```

## Test Requirements

- **TensorFlow**: Most tests require TensorFlow to be installed. Tests will be skipped if TensorFlow is not available.
- **Small Datasets**: Tests use small synthetic datasets (100-120 samples) for fast execution.
- **Time Budgets**: Tests use short time budgets (15-30 seconds) to complete quickly.
- **Temporary Files**: Tests use temporary directories for checkpoints and repositories.

## Notes

- Integration tests may take several minutes to complete due to actual model training.
- Tests are designed to be deterministic where possible (using random seeds).
- Some tests may show warnings about convergence or performance - this is expected with small datasets.
- Tests verify functional correctness, not optimal performance.
