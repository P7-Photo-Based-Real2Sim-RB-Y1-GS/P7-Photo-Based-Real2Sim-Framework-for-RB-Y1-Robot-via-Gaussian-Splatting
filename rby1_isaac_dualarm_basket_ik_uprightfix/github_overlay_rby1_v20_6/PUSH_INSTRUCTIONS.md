# Push instructions for ISAACSIM branch

Repository:
https://github.com/P7-Photo-Based-Real2Sim-RB-Y1-GS/P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot

Branch:
ISAACSIM

## Fresh clone

```bash
cd ~/다운로드

git clone -b ISAACSIM https://github.com/P7-Photo-Based-Real2Sim-RB-Y1-GS/P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot.git
cd P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot

unzip -o /path/to/github_overlay_rby1_v20_6.zip -d .

git status
git add isaac camera scripts tools assets docs .gitignore
git commit -m "Add RB-Y1 webcam STL grasp pipeline for Isaac Sim"
git push origin ISAACSIM
```

## Existing local clone

```bash
cd ~/P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot

git fetch origin
git switch ISAACSIM || git switch -c ISAACSIM origin/ISAACSIM

git pull origin ISAACSIM
unzip -o /path/to/github_overlay_rby1_v20_6.zip -d .

git status
git add isaac camera scripts tools assets docs .gitignore
git commit -m "Add RB-Y1 webcam STL grasp pipeline for Isaac Sim"
git push origin ISAACSIM
```

Do not add `__pycache__/` or `.backup_*` files.
