# Contributing to FedAvg Project

Thank you for your interest in contributing to our Federated Learning implementation! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit bug fixes
- ✨ Add new features
- 🧪 Write tests

## 🚀 Getting Started

### 1. Fork the Repository

Click the "Fork" button at the top right of the repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/MAJOR_PROJECT_1.git
cd MAJOR_PROJECT_1
```

### 3. Set Up Development Environment

**Windows:**
```batch
cd FedAvg
SETUP_EASY.bat
```

**Linux/Mac:**
```bash
cd FedAvg
conda create -n fedavg python=3.9 -y
conda activate fedavg
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install numpy matplotlib -y
pip install wandb pytest
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## 📝 Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

Example:
```python
def calculate_accuracy(predictions, labels):
    """
    Calculate classification accuracy.
    
    Args:
        predictions (torch.Tensor): Model predictions
        labels (torch.Tensor): Ground truth labels
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total
```

### Documentation

- Update README.md if adding new features
- Add comments for complex logic
- Include usage examples

## 🧪 Testing

### Running Tests

```bash
cd FedAvg
pytest tests/
```

### Writing Tests

- Add tests for new features
- Ensure tests pass before submitting PR
- Include edge cases

Example:
```python
def test_federated_sampler():
    """Test federated data sampling"""
    dataset = MNISTDataset(root="../datasets/", train=True)
    sampler = FederatedSampler(dataset, partition_mode="iid", n_clients=10)
    
    # Check all clients have data
    assert len(sampler.dict_users) == 10
    
    # Check data distribution
    total_samples = sum(len(sampler.dict_users[i]) for i in range(10))
    assert total_samples == len(dataset)
```

## 📋 Pull Request Process

### 1. Ensure Quality

- [ ] Code follows style guidelines
- [ ] Tests pass (`pytest tests/`)
- [ ] Documentation is updated
- [ ] No unnecessary files included

### 2. Commit Your Changes

```bash
git add .
git commit -m "Add feature: your feature description"
```

Use clear, descriptive commit messages:
- ✅ "Add Dirichlet partitioning with custom alpha"
- ✅ "Fix CUDA memory leak in client training"
- ❌ "Update code"
- ❌ "Fixed stuff"

### 3. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template:

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] Added new tests (if applicable)

## Screenshots (if applicable)
```

## 🐛 Reporting Bugs

### Before Reporting

- Check existing issues
- Try the latest version
- Verify it's reproducible

### Bug Report Template

```markdown
**Describe the Bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command: `python fed_avg.py --dataset mnist ...`
2. Observe error: ...

**Expected Behavior**
What should happen

**Environment**
- OS: Windows 10 / Ubuntu 20.04 / macOS 12
- Python: 3.9.13
- PyTorch: 1.12.1
- CUDA: 11.6 (if applicable)

**Additional Context**
- Error messages
- Screenshots
- Logs
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this be implemented?

**Alternatives Considered**
Other approaches you've thought about
```

## 📚 Areas Needing Contribution

### High Priority

- [ ] Add more datasets (Fashion-MNIST, SVHN)
- [ ] Implement FedProx algorithm
- [ ] Add differential privacy support
- [ ] Improve documentation with examples

### Medium Priority

- [ ] Add model compression techniques
- [ ] Create Jupyter notebook tutorials
- [ ] Implement FedNova algorithm
- [ ] Add visualization tools

### Good First Issues

- [ ] Fix typos in documentation
- [ ] Add more unit tests
- [ ] Improve error messages
- [ ] Add progress bars

## 🤝 Code Review Process

1. Maintainers review PRs within 1-3 days
2. Feedback and requested changes are discussed
3. Once approved, PR is merged
4. Contributors are added to acknowledgments

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private inquiries

## 🏆 Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

✅ **Positive behavior:**
- Using welcoming language
- Being respectful of differing viewpoints
- Accepting constructive criticism
- Focusing on what's best for the community

❌ **Unacceptable behavior:**
- Trolling or insulting comments
- Personal or political attacks
- Harassment of any kind
- Publishing others' private information

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the FedAvg project! 🎉
