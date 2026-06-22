import pickle


def save_object(obj, filename="data.pickle"):
    """
    Save an object to a file using pickle.

    Args:
        obj (object): The object to save.
        filename (str, optional): The filename to save the object to. Defaults to "data.pickle".
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    """
    Load parameters from a pickle file.

    Args:
        filename (str): The filename to load the parameters from.

    Returns:
        object: The loaded parameters.
    """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object:", ex)
        return None