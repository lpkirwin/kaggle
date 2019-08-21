# kaggle

Basic repo (hopefully) applicable to many kaggle competitions

## To-do

- [ ] Stacking
- [ ] Model metadata

## Environment

### Windows

    conda create --name kaggle python=3.7
    conda activate kaggle
    pip install -r requirements.txt
    python -m ipykernel install --user --name kaggle
    jupyter contrib nbextension install --user

## Kaggle API

See [kaggle API github page](https://github.com/Kaggle/kaggle-api).

`kaggle.json` file from website needs to be placed under `$HOME/.kaggle/`.

Example commands:

```bash
kaggle competitions list  # see existing competitions
kaggle competitions files <competition>  # view data for competition
```

Script to set up a new competition (creates folders and downloads data):

```bash
python new_competition.py <competition>  # need to 'join' on website beforehand
```

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
