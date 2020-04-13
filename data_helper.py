import os
from pathlib import Path

root_path = Path(os.path.dirname(os.path.abspath(__file__)))

def join_root_path(path):
    join_path = os.path.join(root_path, path)
    if not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path

data_folder = os.path.join(str(root_path),'data')
cv_folder = os.path.join(str(root_path), 'data')


file_list = ['australian', 'balance', 'bupa']