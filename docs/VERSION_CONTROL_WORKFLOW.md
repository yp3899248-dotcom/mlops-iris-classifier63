# Version Control Workflow — MLOps Iris Classifier

## 1. Overview

This document describes the Git-based version control workflow used for this Machine Learning project, developed as part of MLOps Lab Experiment 2.

- Repository: mlops-iris-classifier63
- Primary language: Python
- Maintainer: Yash Patil

## 2. Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, deployable code |
| `develop` | Integration branch |
| `feature/*` | Individual features |
| `conflict-demo-*` | Merge conflict demonstration |

**Workflow:**

`feature/* → Pull Request → develop → main`

No direct commits should be made to `main`.

## 3. Commit Convention

Commits follow this format:

`<type>: <short description>`

Examples:

- `feat:` New functionality
- `fix:` Bug fixes
- `docs:` Documentation changes
- `chore:` Configuration or maintenance
- `refactor:` Code restructuring

Example:

`feat: add classification report to training script`

## 4. Merge Conflict Resolution

1. Attempt the merge.
2. Git identifies conflicting files.
3. Open the conflicted file.
4. Locate `<<<<<<<`, `=======`, and `>>>>>>>`.
5. Choose the required change.
6. Remove all conflict markers.
7. Save the file.
8. Run `git add <file>`.
9. Commit the resolved merge.
10. Test the project.

## 5. Pull Request Workflow

1. Create a feature branch.
2. Make the required changes.
3. Commit the changes.
4. Push the branch to GitHub.
5. Create a Pull Request into `develop`.
6. Review the changes.
7. Merge the Pull Request.

## 6. ML Project .gitignore Policy

The following generated or large files should not be committed directly:

- `.venv/`
- `data/*.csv`
- `data/*.parquet`
- `models/*.pkl`
- `models/*.joblib`
- `.ipynb_checkpoints/`

Large datasets and model files can be managed separately using tools such as DVC.

## 7. Verification

- Git version checked
- Feature branch created
- Pull Request created and merged
- `develop` synchronized with GitHub
- Merge conflict created and resolved
- Conflict resolution committed and pushed
- Training script tested successfully