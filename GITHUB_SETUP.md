# 🚀 GitHub Setup Guide

This guide will help you initialize your local Git repository and push it to GitHub.

## Prerequisites

- Git installed: `git --version`
- GitHub account created
- (Optional) GitHub CLI installed: `gh --version`

---

## Step 1: Initialize Local Repository

Open PowerShell in the project directory (`S:\sakshi`) and run:

```powershell
# Navigate to project directory
cd S:\sakshi

# Initialize git repository
git init

# Configure git (if not already done)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files
git add .

# Create initial commit
git commit -m "chore: initial commit - bilingual menu translation system

- Add ML pipeline with MarianMT and IndicBART models
- Add Streamlit and Gradio web interfaces
- Add comprehensive dataset with 500 menu items
- Add CI/CD with GitHub Actions
- Add documentation and contributing guidelines"
```

---

## Step 2: Create GitHub Repository

### Option A: Using GitHub CLI (Recommended)

```powershell
# Login to GitHub (first time only)
gh auth login

# Create repository and push
gh repo create bilingual-menu-translation --public --source=. --remote=origin --push

# Done! Repository is created and code is pushed
```

### Option B: Manual Setup via GitHub Web

1. Go to https://github.com/new
2. Repository name: `bilingual-menu-translation`
3. Description: "AI-powered bilingual menu translation for English to Hindi and Kannada"
4. Visibility: Public
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

Then connect and push:

```powershell
# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/bilingual-menu-translation.git

# Verify remote
git remote -v

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Step 3: Update README with Your Info

After creating the repo, update these placeholders in your files:

1. **README.md**: Replace `YOUR_USERNAME` with your GitHub username
2. **LICENSE**: Replace `[Your Name]` with your actual name
3. **setup.py** and **pyproject.toml**: Update author name and email
4. **README.md**: Update the Author section at the bottom

Quick find & replace:
```powershell
# Use your editor's find & replace feature to replace:
# YOUR_USERNAME -> your-github-username
# [Your Name] -> Your Actual Name
# your.email@example.com -> your.actual.email@example.com
```

Then commit and push the updates:
```powershell
git add .
git commit -m "docs: update author info and GitHub links"
git push
```

---

## Step 4: Verify GitHub Actions

1. Go to your repository on GitHub
2. Click the "Actions" tab
3. You should see the CI workflow run automatically
4. The badge in README.md will update once the workflow completes

---

## Step 5: Add Repository Topics (Optional)

On GitHub repository page:
1. Click the ⚙️ gear icon next to "About"
2. Add topics: `machine-learning`, `nlp`, `translation`, `transformers`, `hindi`, `kannada`, `ocr`, `streamlit`, `pytorch`
3. Save changes

---

## Common Issues

### Authentication Failed
```powershell
# Use GitHub CLI or set up SSH keys
gh auth login

# Or generate a Personal Access Token (classic) with repo scope:
# https://github.com/settings/tokens
```

### Large Files Warning
```powershell
# The .gitignore already excludes model files (saved_model/, *.pt, *.safetensors)
# If you accidentally committed large files:
git rm --cached saved_model/ -r
git commit -m "chore: remove large model files"
git push
```

### Remote Already Exists
```powershell
# Remove and re-add
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/bilingual-menu-translation.git
```

---

## Next Steps

✅ Repository is live on GitHub!

**Recommended:**
1. Add a repository description and website in GitHub settings
2. Enable Discussions for community engagement
3. Star your own repo 😄
4. Share with the community
5. Add repository to your GitHub profile README

**Optional:**
- Set up GitHub Pages for documentation
- Add code coverage badges
- Set up Dependabot for dependency updates
- Create release tags for versions

---

**Need help?** Open an issue or check [GitHub Docs](https://docs.github.com)
