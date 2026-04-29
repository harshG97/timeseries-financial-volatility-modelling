# timeseries-financial-volatility-modelling

## Quick Start: Team Git Workflow

We use feature branches to keep `main` clean. Never commit directly to `main`.

### 1. Start Fresh
Always pull the latest changes before starting new work.
```bash
git checkout main
git pull origin main
```

### 2. Branch Out
Create a branch for your specific task (e.g., `feature/data-split`, `fix/typo`).
```bash
git checkout -b feature/your-branch-name
```

### 3. Work & Commit
Save your changes with clear, descriptive messages.
```bash
git add .
git commit -m "Brief description of what you changed"
```

### 4. Push & PR
Push your branch to GitHub/GitLab and open a Pull Request.
```bash
git push -u origin feature/your-branch-name
```

### 5. Clean Up (After PR is Merged)
Switch back to `main`, pull the updates, and delete your local feature branch.
```bash
git checkout main
git pull origin main
git branch -d feature/your-branch-name
```