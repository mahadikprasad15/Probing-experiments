#!/bin/bash

# GitHub Push Helper Script
# This script will help you push the inoculation-probing repo to GitHub

echo "=================================="
echo "GitHub Push Helper"
echo "=================================="
echo ""
echo "First, you need to create a new repository on GitHub:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: inoculation-probing"
echo "3. Description: Experimental framework for testing inoculation prompts in linear probing"
echo "4. Keep it Public (or Private if you prefer)"
echo "5. DO NOT initialize with README, .gitignore, or license (we already have these)"
echo "6. Click 'Create repository'"
echo ""
echo "After creating the repository, GitHub will show you a page with commands."
echo "Come back here and we'll push your code!"
echo ""
echo "Press Enter when you've created the repository on GitHub..."
read

echo ""
echo "Great! Now I need your GitHub username."
echo "What is your GitHub username? (e.g., mahadikprasad15)"
read GITHUB_USERNAME

echo ""
echo "Perfect! Now pushing to GitHub..."
echo ""

# Navigate to repo
cd /home/user/Probing-experiments/inoculation-probing

# Add remote
REPO_URL="https://github.com/${GITHUB_USERNAME}/inoculation-probing.git"
echo "Adding remote: $REPO_URL"
git remote add origin "$REPO_URL"

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "SUCCESS! 🎉"
    echo "=================================="
    echo ""
    echo "Your repository is now live at:"
    echo "https://github.com/${GITHUB_USERNAME}/inoculation-probing"
    echo ""
else
    echo ""
    echo "=================================="
    echo "Push failed - you may need to authenticate"
    echo "=================================="
    echo ""
    echo "If you see an authentication error, you have two options:"
    echo ""
    echo "Option A: Use Personal Access Token"
    echo "  1. Go to: https://github.com/settings/tokens"
    echo "  2. Generate new token (classic)"
    echo "  3. Select 'repo' scope"
    echo "  4. When prompted for password, use the token instead"
    echo ""
    echo "Option B: Use SSH"
    echo "  1. Run: ssh-keygen -t ed25519 -C 'your_email@example.com'"
    echo "  2. Add SSH key to GitHub: https://github.com/settings/keys"
    echo "  3. Change remote URL:"
    echo "     git remote set-url origin git@github.com:${GITHUB_USERNAME}/inoculation-probing.git"
    echo "  4. Try push again: git push -u origin main"
    echo ""
fi
