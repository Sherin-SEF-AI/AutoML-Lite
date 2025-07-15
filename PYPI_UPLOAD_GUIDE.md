# PyPI Upload Guide for AutoML Lite

## 🚀 Ready for PyPI Upload!

Your AutoML Lite package has been successfully built and validated. Here's everything you need to know for uploading to PyPI.

## 📦 Package Information

- **Package Name**: `automl-lite`
- **Version**: `0.1.0`
- **Author**: sherin joseph roy
- **Email**: sherin.joseph2217@gmail.com
- **Repository**: https://github.com/Sherin-SEF-AI/AutoML-Lite

## 📁 Built Files

The following files have been created in the `dist/` directory:
- `automl_lite-0.1.0.tar.gz` (Source distribution)
- `automl_lite-0.1.0-py3-none-any.whl` (Wheel distribution)

## 🔧 Upload Commands

### 1. Upload to Test PyPI (Recommended First)

```bash
# Upload to Test PyPI for testing
twine upload --repository testpypi dist/*

# Install from Test PyPI to verify
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ automl-lite
```

### 2. Upload to Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*
```

## 🔐 PyPI Account Setup

### 1. Create PyPI Account
- Go to https://pypi.org/account/register/
- Create an account with your email: `sherin.joseph2217@gmail.com`

### 2. Create Test PyPI Account (Optional)
- Go to https://test.pypi.org/account/register/
- Create a separate account for testing

### 3. Configure Credentials
When prompted during upload, enter your PyPI username and password.

## 📋 Pre-Upload Checklist

✅ **Package Configuration**
- [x] Author information updated
- [x] Repository URLs corrected
- [x] Dependencies properly specified
- [x] License file included

✅ **Build Validation**
- [x] Package builds successfully
- [x] Twine validation passes
- [x] No critical warnings

✅ **Documentation**
- [x] README.md comprehensive
- [x] API documentation complete
- [x] Examples provided
- [x] Installation instructions clear

## 🎯 Post-Upload Steps

### 1. Verify Installation
```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the package
pip install automl-lite

# Test the CLI
automl-lite --help
```

### 2. Test Basic Functionality
```python
from automl_lite import AutoMLite
import pandas as pd
import numpy as np

# Create sample data
X = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
y = np.random.randint(0, 2, 100)

# Test AutoML
automl = AutoMLite()
automl.fit(X, y)
predictions = automl.predict(X)
print("Success!")
```

### 3. Update Repository
- Push all changes to GitHub
- Update repository description
- Add PyPI badge to README

## 🏷️ PyPI Badge

Add this badge to your README.md after successful upload:

```markdown
[![PyPI version](https://badge.fury.io/py/automl-lite.svg)](https://badge.fury.io/py/automl-lite)
```

## 📈 Version Management

For future releases:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or "0.2.0" for minor release
   ```

2. **Update CHANGELOG.md** with new features/fixes

3. **Rebuild and upload**:
   ```bash
   rm -rf dist/ build/ src/*.egg-info/
   python -m build
   twine upload dist/*
   ```

## 🆘 Troubleshooting

### Common Issues

1. **Authentication Error**
   - Ensure you're using the correct PyPI credentials
   - Check if you have 2FA enabled

2. **Package Name Conflict**
   - The name `automl-lite` should be available
   - If not, consider `automl-lite-ml` or similar

3. **Upload Rejected**
   - Check for any remaining linter errors
   - Ensure all dependencies are available on PyPI

### Support

If you encounter issues:
- Check PyPI documentation: https://packaging.python.org/
- Review the build logs for warnings
- Test installation in a clean environment

## 🎉 Success!

Once uploaded, users can install your package with:
```bash
pip install automl-lite
```

And use it with:
```python
from automl_lite import AutoMLite
# Your AutoML journey begins!
```

---

**Good luck with your PyPI upload! 🚀** 