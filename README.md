# My Project

cd ~/Documents/NW_methylome
git status
if needed:
git add .
git commit -m "Describe changes"
git push

or git pull

Source .venv/bin/activate
(NW_methylome) squirrel@squirrels-iMac NW_methylome %

python --version
Python 3.11.x

jupyter lab

Kernel → Change Kernel → Python 3.11 (NW_methylome)

inside notebook
import sys
sys.executable

/Users/squirrel/Documents/NW_methylome/.venv/bin/python

Project structure:
NW_methylome/
├── README.md
├── notebooks/
│   └── analysis.ipynb
├── scripts/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── requirements.txt
├── .gitignore
└── .venv/        ← do not commit

uv pip install torch lightning numpy pandas scikit-learn matplotlib seaborn
uv pip freeze > requirements.txt

git status

git add .

git commit -m "Describe today's work"

git push



