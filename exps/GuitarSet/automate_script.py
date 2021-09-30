import subprocess
from pathlib import Path


if __name__ == '__main__':

  
    script_path = Path(Path.cwd(), 'GuitarSetTest.py')
    constants_path = Path(Path.cwd(), 'constants.ini')
    workspace_path = Path(Path.cwd() )
    for dataset in ['mix', 'mic']:
        for train_mode in ['1Fret', '2FretA', '2FretB', '3Fret']:
            subprocess.run(['python', script_path, constants_path, 
                                workspace_path, '--dataset', dataset,
                                    '--train_mode', train_mode])