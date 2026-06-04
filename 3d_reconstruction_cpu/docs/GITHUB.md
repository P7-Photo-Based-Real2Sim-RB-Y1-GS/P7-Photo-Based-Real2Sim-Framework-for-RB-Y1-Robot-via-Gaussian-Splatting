# GitHub Upload Guide

This folder is ready to be uploaded as a GitHub repository.

## 1. Initialize Git

```powershell
cd <YOUR_REPOSITORY_PATH>
git init
git add README.md requirements.txt reconstruct_rgbd_object.py run_reconstruction.ps1 .gitignore .gitattributes docs
git commit -m "Add RGB-D turntable reconstruction pipeline"
```

## 2. Create a GitHub Repository

Create an empty repository on GitHub. Do not add a README from GitHub because this project already has one.

## 3. Connect and Push

Replace `<YOUR_GITHUB_REPO_URL>` with your repository URL.

```powershell
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

## Notes

- Generated reconstruction assets such as `.stl`, `.glb`, `.ply`, raw RGB-D datasets, and `.venv` are ignored by `.gitignore`.
- If you want to share generated 3D files, upload them separately through GitHub Releases or cloud storage.
- Large raw captures such as `.db3`, `.bag`, or `.mcap` should not be committed to normal Git history.
