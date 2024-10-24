import pathlib
import iris


def test_files():
    result = iris.response.files()
    assert len(result) > 0
    for r in result:
        assert isinstance(r, pathlib.Path)
