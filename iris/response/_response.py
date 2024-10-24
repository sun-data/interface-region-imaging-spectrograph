import pathlib

__all__ = [
    "files",
]


def files() -> list[pathlib.Path]:
    """
    A list of the IDL ``.sav`` files storing the IRIS instrument response.
    """

    directory = pathlib.Path(__file__).parent

    result = sorted(directory.glob("*.geny"))

    return result
