# PyPI Release Guide for AutoML Lite v0.2.0

## Package Ready for PyPI! 🚀

Your package has been successfully built and is ready for publication to PyPI.

### Build Status ✅
- **Version**: 0.2.0
- **Wheel**: `automl_lite-0.2.0-py3-none-any.whl` (172K)
- **Source**: `automl_lite-0.2.0.tar.gz` (193K)
- **Location**: `dist/` directory

### What's Included
- Complete NAS implementation (13 modules)
- All core AutoML features
- CLI interface with NAS commands
- Comprehensive documentation
- 32 passing tests
- Full backward compatibility

---

## Publishing to PyPI

### Prerequisites

1. **Install twine** (if not already installed):
```bash
pip install twine
```

2. **Create PyPI account** (if you don't have one):
   - Go to https://pypi.org/account/register/
   - Verify your email address

3. **Create API token** (recommended over password):
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token with scope "Entire account"
   - Save the token securely (you'll only see it once)

---

## Step-by-Step Publishing

### Step 1: Verify the Build

Check that the distribution files are correct:

```bash
twine check dist/*
```

Expected output:
```
Checking dist/automl_lite-0.2.0-py3-none-any.whl: PASSED
Checking dist/automl_lite-0.2.0.tar.gz: PASSED
```

### Step 2: Test on TestPyPI (Recommended)

Before publishing to the real PyPI, test on TestPyPI:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

Then test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ automl-lite==0.2.0
```

### Step 3: Publish to PyPI

Once you've verified everything works on TestPyPI:

```bash
# Upload to PyPI
twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

### Step 4: Verify Publication

After successful upload:

1. Check your package page: https://pypi.org/project/automl-lite/
2. Test installation:
```bash
pip install automl-lite==0.2.0
```

---

## Alternative: Using .pypirc

To avoid entering credentials each time, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

Then you can upload without prompts:
```bash
twine upload dist/*
```

---

## Post-Publication Checklist

### 1. Create GitHub Release

```bash
# Create and push a tag
git tag -a v0.2.0 -m "Release v0.2.0: Neural Architecture Search"
git push origin v0.2.0
```

Then on GitHub:
- Go to Releases → Draft a new release
- Choose tag: v0.2.0
- Title: "v0.2.0 - Neural Architecture Search"
- Description: Copy from CHANGELOG.md
- Attach dist files (optional)
- Publish release

### 2. Update Documentation

- Update README badges with new version
- Update installation instructions
- Add release announcement

### 3. Announce the Release

Consider announcing on:
- GitHub Discussions
- Twitter/X
- LinkedIn
- Reddit (r/MachineLearning, r/Python)
- Dev.to or Medium blog post

---

## Version Management

### For Future Releases

1. **Update version** in `pyproject.toml`:
```toml
version = "0.2.1"  # or 0.3.0, 1.0.0, etc.
```

2. **Update CHANGELOG.md** with new changes

3. **Clean and rebuild**:
```bash
rm -rf dist/ build/ *.egg-info
python3 -m build
```

4. **Commit and tag**:
```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: Bump version to X.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main --tags
```

5. **Publish to PyPI**:
```bash
twine upload dist/*
```

---

## Semantic Versioning Guide

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.2.1): Bug fixes, backward compatible

Current version: **0.2.0** (Minor release with NAS feature)

Suggested next versions:
- **0.2.1**: Bug fixes
- **0.3.0**: Next feature addition
- **1.0.0**: Stable production release

---

## Troubleshooting

### Issue: "File already exists"

If you get this error, you're trying to upload a version that already exists on PyPI.

**Solution**: Bump the version number in `pyproject.toml` and rebuild.

### Issue: "Invalid distribution"

**Solution**: Run `twine check dist/*` to see what's wrong.

### Issue: "Authentication failed"

**Solutions**:
1. Make sure username is `__token__` (not your PyPI username)
2. Verify your API token is correct
3. Check token hasn't expired
4. Ensure token has correct scope

### Issue: "Package name already taken"

If `automl-lite` is taken (unlikely since you're the owner):
- Contact PyPI support
- Or choose a different name in `pyproject.toml`

---

## Quick Reference Commands

```bash
# Check build
twine check dist/*

# Test on TestPyPI
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*

# Create git tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# Install from PyPI
pip install automl-lite==0.2.0
```

---

## Package Statistics

After publication, you can track:
- Downloads: https://pypistats.org/packages/automl-lite
- Package health: https://snyk.io/advisor/python/automl-lite
- Dependencies: https://libraries.io/pypi/automl-lite

---

## Support

If you encounter issues:
1. Check PyPI documentation: https://packaging.python.org/
2. PyPI help: https://pypi.org/help/
3. Contact: pypi-admins@python.org

---

**Ready to publish!** Run `twine upload dist/*` when you're ready to go live. 🎉
