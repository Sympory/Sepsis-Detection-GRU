# Contributing to Sepsis Detection GRU

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Relevant error messages/logs

### Suggesting Enhancements

We welcome feature requests! Please open an issue with:
- Clear description of the enhancement
- Use case / motivation
- Possible implementation approach
- Any relevant examples or references

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/sepsis-detection-gru.git
   cd sepsis-detection-gru
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, commented code
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Test the web app
   python app.py
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

   Use commit prefixes:
   - `Add:` New feature
   - `Fix:` Bug fix
   - `Update:` Improvement to existing feature
   - `Docs:` Documentation changes
   - `Refactor:` Code restructuring
   - `Test:` Adding/updating tests

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub with:
   - Clear title and description
   - Reference any related issues
   - Screenshots/demos if applicable

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings for functions and classes
- Keep functions focused and concise

**Example:**
```python
def predict_sepsis_risk(vital_signs: dict, window_size: int = 6) -> float:
    """
    Predict sepsis risk from patient vital signs.
    
    Args:
        vital_signs: Dictionary of vital sign measurements
        window_size: Lookback window in hours (default: 6)
    
    Returns:
        float: Sepsis risk probability [0, 1]
    
    Raises:
        ValueError: If vital_signs is empty or invalid
    """
    # Implementation
    pass
```

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

### Documentation

- Update README.md if adding features
- Update docstrings
- Add comments for complex logic
- Update CHANGELOG.md

## Project Areas for Contribution

### High Priority

- [ ] Multi-institutional dataset validation
- [ ] SHAP/LIME explainability integration
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Unit test coverage

### Medium Priority

- [ ] Attention mechanism implementation
- [ ] Bidirectional GRU variant
- [ ] Model ensemble methods
- [ ] Feature engineering (temporal derivatives)
- [ ] Enhanced web UI (charts, alerts)

### Documentation

- [ ] API documentation (Swagger/OpenAPI)
- [ ] Tutorial notebooks (Jupyter)
- [ ] Video tutorials
- [ ] Translation to other languages
- [ ] Architecture diagrams

### Infrastructure

- [ ] PostgreSQL database support
- [ ] Redis caching layer
- [ ] Kubernetes deployment manifests
- [ ] Monitoring and logging (Prometheus, Grafana)
- [ ] Load testing

## Questions?

Feel free to open a discussion or reach out via:
- GitHub Issues
- Email: [your-email]
- Discussion forum

## Code of Conduct

Be respectful and professional in all interactions. We aim to create a welcoming and inclusive environment for all contributors.

---

**Thank you for making healthcare AI better! 🏥💙**
