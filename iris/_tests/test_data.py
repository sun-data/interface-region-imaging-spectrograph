from __future__ import annotations
from typing import Sequence
import pytest
import pathlib
import astropy.time
import iris


_obsid_b2 = 3893012099


@pytest.mark.parametrize("time_start", [None])
@pytest.mark.parametrize("time_stop", [None])
@pytest.mark.parametrize("description", [""])
@pytest.mark.parametrize("obs_id", [None, _obsid_b2])
@pytest.mark.parametrize("limit", [5])
def test_query_hek(
    time_start: None | astropy.time.Time,
    time_stop: None | astropy.time.Time,
    description: str,
    obs_id: None | str,
    limit: int,
):
    result = iris.data.query_hek(
        time_start=time_start,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        limit=limit,
    )
    assert isinstance(result, str)


@pytest.mark.parametrize("time_start", [None])
@pytest.mark.parametrize("time_stop", [None])
@pytest.mark.parametrize("description", [""])
@pytest.mark.parametrize("obs_id", [None, _obsid_b2])
@pytest.mark.parametrize("limit", [5])
@pytest.mark.parametrize("spectrograph", [True])
@pytest.mark.parametrize("sji", [True])
@pytest.mark.parametrize("deconvolved", [True])
def test_urls_hek(
    time_start: None | astropy.time.Time,
    time_stop: None | astropy.time.Time,
    description: str,
    obs_id: None | str,
    limit: int,
    spectrograph: bool,
    sji: bool,
    deconvolved: bool,
):
    result = iris.data.urls_hek(
        time_start=time_start,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        limit=limit,
        spectrograph=spectrograph,
        sji=sji,
        deconvolved=deconvolved,
    )
    assert isinstance(result, list)
    assert len(result) > 0
    for url in result:
        assert isinstance(url, str)


@pytest.mark.parametrize(
    argnames="urls",
    argvalues=[
        iris.data.urls_hek(
            obs_id=_obsid_b2,
            limit=1,
            sji=False,
        ),
    ],
)
@pytest.mark.parametrize("directory", [None])
@pytest.mark.parametrize("overwrite", [False])
def test_download(
    urls: list[str],
    directory: None | pathlib.Path,
    overwrite: bool,
):
    result = iris.data.download(
        urls=urls,
        directory=directory,
        overwrite=overwrite,
    )
    assert isinstance(result, list)
    assert len(urls) == 1
    for file in result:
        assert file.exists()


@pytest.mark.parametrize(
    argnames="archives",
    argvalues=[
        iris.data.download(
            urls=iris.data.urls_hek(
                obs_id=_obsid_b2,
                limit=1,
                sji=False,
            )
        )
    ],
)
@pytest.mark.parametrize("directory", [None])
@pytest.mark.parametrize("overwrite", [False])
def test_decompress(
    archives: Sequence[pathlib.Path],
    directory: pathlib.Path,
    overwrite: bool,
):
    result = iris.data.decompress(
        archives=archives,
        directory=directory,
        overwrite=overwrite,
    )
    assert isinstance(result, list)
    for file in result:
        assert file.exists()
        assert file.suffix == ".fits"
