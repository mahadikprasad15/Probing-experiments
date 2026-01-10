# How to Push inoculation-probing to GitHub

## Step 1: Create Repository on GitHub

Go to: https://github.com/new

- Repository name: `inoculation-probing`
- Description: `Experimental framework for testing inoculation prompts in linear probing`
- **DO NOT** initialize with README, .gitignore, or license

Click "Create repository"

## Step 2: Run These Commands

**Replace `YOUR_USERNAME` with your actual GitHub username** (probably `mahadikprasad15`):

```bash
cd inoculation-probing

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/inoculation-probing.git

# Push to GitHub
git push -u origin main
```

## Step 3: Authenticate

When prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (NOT your GitHub password!)

### How to get a Personal Access Token:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name it: `inoculation-probing-push`
4. Check the **`repo`** box (full control of repositories)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as your password when git asks

## Done!

After pushing, visit:
`https://github.com/YOUR_USERNAME/inoculation-probing`

You should see all your code there!

## Quick Command Summary

```bash
cd inoculation-probing
git remote add origin https://github.com/YOUR_USERNAME/inoculation-probing.git
git push -u origin main
```
