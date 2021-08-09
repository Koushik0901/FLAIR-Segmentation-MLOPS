import os

dirs = [
    'data/raw',
    'data/preprocessed',
    'notebooks',
    'saved_models',
    'src'
]

for _dir in dirs:
    os.makedirs(_dir, exist_ok=True)
    with open(os.path.join(_dir, '.gitkeep'), 'w') as f:
        ...


files = [
    'dvc.yaml',
    'params.yaml',
    '.gitignore',
    'src/__init__.py',
    'README.md'
]

for _file in files:
    with open(_file, 'w') as f:
        ...
