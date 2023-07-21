import importlib
from typing import Any


def import_object_from_path(path: str) -> Any:
    """Dynamically imports a Python object from the given path.

    This function takes a string representing a path to a Python object
    (e.g., a class, function, or variable) and returns that object. It
    allows dynamically importing objects based on a specified path.

    Args:
        path (str): The path to the Python object in the format "module.submodule.object".

    Returns:
        Any: The Python object specified by the given path.

    Raises:
        ImportError: If the specified object cannot be imported or retrieved.
            This may occur due to incorrect module or object names or if the
            module or object does not exist.

    Example:
        >>> obj = import_object_from_path("os.path.join")
        >>> print(obj)
        <function join>
    """
    try:
        module_path, object_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, object_name)
        return obj
    except (ImportError, AttributeError) as error:
        raise ImportError(f"Could not import object from path '{path}': {error}") from error
