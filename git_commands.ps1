# Quick Git Setup - Copy and paste these commands

# Step 1: Initialize and commit
cd S:\sakshi
git init
git add .
git commit -m "chore: initial commit - bilingual menu translation system"

# Step 2A: Using GitHub CLI (easiest)
gh auth login
gh repo create bilingual-menu-translation --public --source=. --remote=origin --push

# Step 2B: Manual (if not using GitHub CLI)
# First create repo on https://github.com/new
# Then run:
# git remote add origin https://github.com/YOUR_USERNAME/bilingual-menu-translation.git
# git branch -M main
# git push -u origin main

# Done! 🎉
