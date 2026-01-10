# How to Push to GitHub

## Quick Method: Use the Helper Script

```bash
cd /home/user/Probing-experiments/inoculation-probing
./push_to_github.sh
```

The script will guide you through the process step by step.

---

## Manual Method: Step-by-Step Instructions

### Step 1: Create a New Repository on GitHub

1. Go to: https://github.com/new
2. Fill in the details:
   - **Repository name**: `inoculation-probing`
   - **Description**: `Experimental framework for testing inoculation prompts in linear probing`
   - **Visibility**: Public (or Private if you prefer)
   - **Important**: DO NOT check any of these boxes:
     - ❌ Add a README file
     - ❌ Add .gitignore
     - ❌ Choose a license

   (We already have these files in your local repo)

3. Click **"Create repository"**

### Step 2: Copy Your GitHub Username

After creating the repo, you'll see a page with setup instructions. Note your GitHub username (e.g., `mahadikprasad15`).

### Step 3: Add Remote and Push

**Replace `YOUR_USERNAME` with your actual GitHub username:**

```bash
cd /home/user/Probing-experiments/inoculation-probing

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/inoculation-probing.git

# Push to GitHub
git push -u origin main
```

### Step 4: Authenticate

When you run `git push`, you'll be asked for authentication. You have two options:

#### Option A: Personal Access Token (Recommended for HTTPS)

1. **Generate a token**:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Give it a name: `inoculation-probing-push`
   - Select scopes: Check **`repo`** (this gives full control of private repositories)
   - Click "Generate token"
   - **Copy the token immediately** (you won't see it again!)

2. **When prompted**:
   - Username: Your GitHub username
   - Password: **Paste your Personal Access Token** (not your GitHub password!)

#### Option B: SSH Key (Advanced)

1. **Generate SSH key** (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default location
   # Press Enter for no passphrase (or set one if you prefer)
   ```

2. **Copy your public key**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. **Add to GitHub**:
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Title: `inoculation-probing-machine`
   - Key: Paste the contents from step 2
   - Click "Add SSH key"

4. **Change remote URL to SSH**:
   ```bash
   git remote set-url origin git@github.com:YOUR_USERNAME/inoculation-probing.git
   git push -u origin main
   ```

---

## Verify Success

After pushing, you should see:

```
Enumerating objects: 23, done.
Counting objects: 100% (23/23), done.
Delta compression using up to 8 threads
Compressing objects: 100% (21/21), done.
Writing objects: 100% (23/23), 50.23 KiB | 5.02 MiB/s, done.
Total 23 (delta 1), reused 0 (delta 0), pack-reused 0
To https://github.com/YOUR_USERNAME/inoculation-probing.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

Then visit: `https://github.com/YOUR_USERNAME/inoculation-probing`

You should see all your files there! 🎉

---

## Troubleshooting

### "remote origin already exists"

```bash
# Remove existing remote and add again
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/inoculation-probing.git
git push -u origin main
```

### "Authentication failed"

If using HTTPS:
- Make sure you're using a **Personal Access Token**, not your password
- Token must have `repo` scope

If using SSH:
- Run `ssh -T git@github.com` to test connection
- Should see: "Hi YOUR_USERNAME! You've successfully authenticated..."

### "Updates were rejected"

If the GitHub repo already has commits:
```bash
git pull origin main --rebase
git push -u origin main
```

---

## Next Steps After Pushing

Once your code is on GitHub:

1. **Update README** with your GitHub username:
   - Edit the URLs in README.md to point to your repo
   - Commit and push changes

2. **Share the Colab notebook**:
   - Open: `https://colab.research.google.com/github/YOUR_USERNAME/inoculation-probing/blob/main/notebooks/mvp_experiment.ipynb`
   - This creates a direct Colab link to your notebook!

3. **Set up GitHub Pages** (optional):
   - Settings → Pages → Source: main branch
   - Your README will be visible as a webpage

---

**Need help?** Let me know which step you're stuck on!
