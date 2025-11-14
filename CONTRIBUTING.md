# Contributing to AutoML Lite

Thank you for your interest in contributing to AutoML Lite! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Code of Conduct](#code-of-conduct)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of machine learning concepts
- Familiarity with Python development

### Fork and Clone

1. **Fork the repository**
   - Go to [https://github.com/Sherin-SEF-AI/AutoML-Lite](https://github.com/Sherin-SEF-AI/AutoML-Lite)
   - Click the "Fork" button in the top right

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AutoML-Lite.git
   cd AutoML-Lite
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/Sherin-SEF-AI/AutoML-Lite.git
   ```

## Development Setup

### Environment Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Development Tools

The project uses several development tools:

- **pytest**: Testing framework
- **flake8**: Code linting
- **black**: Code formatting
- **mypy**: Type checking
- **pre-commit**: Git hooks for code quality

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (black default)
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all functions and methods
- **Imports**: Organized with isort

### Code Formatting

Run code formatting before committing:

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting

Check code quality:

```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=automl_lite

# Run specific test file
pytest tests/test_automl.py

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Writing Tests

#### Test Structure

```python
# tests/test_example.py
import pytest
from automl_lite import AutoMLite

class TestAutoMLite:
    """Test cases for AutoMLite class."""
    
    def test_basic_functionality(self):
        """Test basic AutoML functionality."""
        # Arrange
        automl = AutoMLite()
        
        # Act
        result = automl.some_method()
        
        # Assert
        assert result is not None
    
    @pytest.mark.parametrize("input_value,expected", [
        (1, True),
        (0, False),
        (-1, False)
    ])
    def test_parameterized_function(self, input_value, expected):
        """Test function with different parameters."""
        # Test implementation
        pass
```

#### Test Guidelines

- **Test naming**: Use descriptive test names
- **Test structure**: Follow Arrange-Act-Assert pattern
- **Test isolation**: Each test should be independent
- **Test coverage**: Aim for high test coverage
- **Mocking**: Use mocks for external dependencies

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

## Documentation

### Code Documentation

#### Docstrings

Use Google style docstrings:

```python
def train_model(self, data: pd.DataFrame, target: str) -> None:
    """Train the AutoML model on the provided data.
    
    Args:
        data: Training data as a pandas DataFrame.
        target: Name of the target column.
        
    Raises:
        ValueError: If data is empty or target column not found.
        RuntimeError: If training fails.
        
    Example:
        >>> automl = AutoMLite()
        >>> automl.train_model(data, 'target_column')
    """
    pass
```

#### Type Hints

Always include type hints:

```python
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

def process_data(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> np.ndarray:
    """Process data and return numpy array."""
    pass
```

### Documentation Guidelines

1. **Keep it simple**: Write clear, concise documentation
2. **Include examples**: Provide working code examples
3. **Update regularly**: Keep documentation in sync with code
4. **Use proper formatting**: Follow markdown conventions

## Pull Request Process

### Before Submitting

1. **Update your fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation
   - Update changelog if needed

4. **Test your changes**
   ```bash
   # Run all tests
   pytest
   
   # Run linting
   flake8 src/ tests/
   mypy src/
   
   # Run formatting checks
   black --check src/ tests/
   isort --check-only src/ tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(automl): add support for custom algorithms
fix(cli): resolve argument parsing issue
docs(readme): update installation instructions
```

### Submitting the PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template

3. **PR Template**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Documentation
- [ ] Code documented
- [ ] README updated
- [ ] API docs updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added and passing
- [ ] Documentation updated
```

### PR Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Address feedback** and make changes
4. **Squash commits** if requested
5. **Merge** when approved

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment information**
   - Operating system and version
   - Python version
   - AutoML Lite version
   - Dependencies versions

2. **Reproduction steps**
   - Clear, step-by-step instructions
   - Minimal code example
   - Expected vs actual behavior

3. **Error messages**
   - Full error traceback
   - Any relevant logs

### Issue Template

```markdown
## Bug Description
Brief description of the bug

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- AutoML Lite: [e.g., 1.0.0]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Error Messages
```
Full error traceback here
```

## Additional Information
Any other relevant information
```

## Feature Requests

### Feature Request Guidelines

1. **Check existing issues** first
2. **Describe the problem** you're solving
3. **Propose a solution** with examples
4. **Consider implementation** complexity
5. **Discuss alternatives** if applicable

### Feature Request Template

```markdown
## Problem Statement
Describe the problem you're trying to solve

## Proposed Solution
Describe your proposed solution

## Use Cases
Provide specific use cases and examples

## Implementation Considerations
Any implementation details or concerns

## Alternatives Considered
Other approaches you've considered
```

## Code of Conduct

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be patient** with newcomers
- **Be helpful** and supportive

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Any conduct inappropriate in a professional setting

### Enforcement

Violations will be addressed by project maintainers. Contact sherin@deepmost.ai for concerns.

## Getting Help

### Resources

- **Documentation**: [GitHub Wiki](https://github.com/Sherin-SEF-AI/AutoML-Lite/wiki)
- **Issues**: [GitHub Issues](https://github.com/Sherin-SEF-AI/AutoML-Lite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sherin-SEF-AI/AutoML-Lite/discussions)
- **Email**: sherin@deepmost.ai
- **Website**: [sherinjosephroy.link](https://sherinjosephroy.link)

### Community Guidelines

1. **Be patient** with responses
2. **Search existing issues** before posting
3. **Provide context** in your questions
4. **Help others** when you can
5. **Follow the code of conduct**

## Recognition

### Contributors

Contributors will be recognized in:

- **README.md**: List of contributors
- **CHANGELOG.md**: Credit for contributions
- **GitHub**: Contributor statistics
- **Documentation**: Credit for significant contributions

### Types of Contributions

We welcome various types of contributions:

- **Code**: Bug fixes, new features, improvements
- **Documentation**: Guides, tutorials, API docs
- **Testing**: Test cases, bug reports
- **Design**: UI/UX improvements, graphics
- **Community**: Helping others, organizing events

## License

By contributing to AutoML Lite, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AutoML Lite! Your contributions help make this project better for everyone.

**Made with ❤️ by Sherin Joseph Roy** 