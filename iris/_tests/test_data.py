from __future__ import annotations
import pytest
import pathlib
import urlpath
import astropy.time
import iris


@pytest.mark.parametrize("time_start", [None])
@pytest.mark.parametrize("time_stop", [None])
@pytest.mark.parametrize("description", [""])
@pytest.mark.parametrize("obs_id", [None, 3882010194])
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
@pytest.mark.parametrize("obs_id", [None, 3882010194])
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
        assert isinstance(url, urlpath.URL)


@pytest.mark.parametrize(
    argnames="urls",
    argvalues=[
        iris.data.urls_hek(
            obs_id=3882010194,
            limit=1,
            spectrograph=False,
        ),
    ],
)
@pytest.mark.parametrize("directory", [None])
@pytest.mark.parametrize("overwrite", [False])
def test_download(
    urls: list[urlpath.URL],
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
