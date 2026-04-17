import yaml
from typing import List, Dict, Any, Optional

def read_yaml_file(filepath: str) -> str:
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
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def find_specific_variables(
    features_dict: Dict[str, Any], 
    specific_key: str, 
    specific_value: Optional[str] = None
) -> List[str]:
    """
    Return a list of feature names whose metadata contains a given key, and optionally
    whose value for that key equals a specific value.

    Parameters
    ----------
    features_dict : dict
        Mapping of feature name -> metadata dict.
    specific_key : str
        The metadata key to search for (e.g., 'type').
    specific_value : any, optional
        If provided, only features whose metadata value for specific_key equals this
        value will be returned.

    Returns
    -------
    list
        Feature names that satisfy the condition.
    """
    keys_with_specific_key = []

    for k, sub_dict in features_dict.items():
        if isinstance(sub_dict, dict):
            if specific_key in sub_dict:
                if specific_value:
                    if sub_dict[specific_key] == specific_value:
                        keys_with_specific_key.append(k)
                else:
                    keys_with_specific_key.append(k)
    return keys_with_specific_key



def get_features_attribute(
    features: Dict[str, Any], 
    attribute: str
) -> Dict[str, str]:
    """
    Build a mapping of feature name to a given attribute found in its metadata.

    Parameters
    ----------
    features : dict
        Mapping of feature name -> metadata dict.
    attribute : str
        The metadata attribute to extract (e.g., 'type', 'string_to_fill').

    Returns
    -------
    dict
        Mapping feature name -> attribute value for features that define attribute.
    """
    features_to_group = {}

    for k, v in features.items():
        if attribute in v:
            features_to_group[k] = v[attribute]
    return features_to_group