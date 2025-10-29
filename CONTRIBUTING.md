# Contributing

Thanks for your interest in contributing! Please follow these steps to make contributions easy to review and merge.

1. Fork the repository and create a branch from `main` with a descriptive name (e.g., `feature/add-ocr`).
2. Run the environment setup:
   - On Windows (PowerShell):
     ```powershell
     # create and activate venv
     python -m venv .venv; .\.venv\Scripts\Activate.ps1
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
3. Run tests (if any):
   ```powershell
   pytest -q
   ```
4. Follow code style: keep code readable, add docstrings and comments.
5. Open a pull request against `main`. Describe changes and link any related issues.

Maintainers will review and request changes if necessary. Thanks!
