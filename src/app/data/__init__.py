import os 
import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_features(name: str, folder: str = None) -> str:
    """
    Read the contents of an YAML file and return it as a string.

    Parameters
    ----------
    filepath : str
        The path to the YAML file.

    Returns
    -------
    str
        The content of the YAML file as a string.
    """

    if folder:
        path = os.path.join(base_dir, folder, name)
    else:
        path = os.path.join(base_dir, name)
    
    with open(path, "r") as file:
        return yaml.safe_load(file)