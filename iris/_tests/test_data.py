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
@pytest.mark.parametrize("nrt", [False, True])
def test_query_hek(
    time_start: None | astropy.time.Time,
    time_stop: None | astropy.time.Time,
    description: str,
    obs_id: None | str,
    limit: int,
    nrt: bool,
):
    result = iris.data.query_hek(
        time_start=time_start,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        limit=limit,
        nrt=nrt,
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
                time_stop=astropy.time.Time("2024-01-01"),
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


@pytest.mark.parametrize(
    argnames="time,expected",
    argvalues=[
        ("2019-09-30T17:19:30", "2019-09-30T17:20:00"),
        ("2019-09-30T17:19:00", "2019-09-30T17:19:00"),
        ("2019-09-30T17:59:00.5", "2019-09-30T18:00:00"),
    ],
)
def test_ceil_minute(time: str, expected: str):
    result = iris.data._ceil_minute(astropy.time.Time(time))
    assert result == astropy.time.Time(expected)


def test_query_hek_stop():
    """The stop time is rounded up rather than cut to the minute."""
    result = iris.data.query_hek(
        time_start=astropy.time.Time("2019-09-30T17:18:30"),
        time_stop=astropy.time.Time("2019-09-30T17:19:30"),
    )
    assert "startTime=2019-09-30T17:18&" in result
    assert "stopTime=2019-09-30T17:20&" in result


def test_urls_hek_partial_minute():
    """
    An observation which begins during the last partial minute of the
    search period is found.
    """
    result = iris.data.urls_hek(
        time_start=astropy.time.Time("2019-09-30T17:18:30"),
        time_stop=astropy.time.Time("2019-09-30T17:19:30"),
        spectrograph=False,
    )
    assert any("20190930_171911_3604109624" in url for url in result)


_url_final = (
    "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2026/10/06/"
    "20261006_155741_3402506433/iris_l2_20261006_155741_3402506433_SJI_1400_t000.fits.gz"
)
_url_nrt = _url_final.replace("level2_compressed", "level2_nrt_compressed")
_url_nrt_other = _url_nrt.replace("SJI_1400", "SJI_2796")


@pytest.mark.parametrize(
    argnames="url,expected",
    argvalues=[
        (_url_final, False),
        (_url_nrt, True),
    ],
)
def test_is_nrt(url: str, expected: bool):
    assert iris.data._is_nrt(url) == expected


@pytest.mark.parametrize(
    argnames="urls,expected",
    argvalues=[
        ([_url_final], [_url_final]),
        ([_url_nrt], [_url_nrt]),
        ([_url_final, _url_nrt], [_url_final]),
        ([_url_nrt, _url_final], [_url_final]),
        ([_url_final, _url_nrt, _url_nrt_other], [_url_final, _url_nrt_other]),
        ([_url_final, _url_final], [_url_final]),
    ],
)
def test_prefer_final(urls: list[str], expected: list[str]):
    assert iris.data._prefer_final(urls) == expected


def test_download_nrt(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """The NRT and final files of an observation are kept apart."""

    class Response:
        def __init__(self, url: str):
            self.content = url.encode()

    def get(url: str, *args: object, **kwargs: object) -> Response:
        return Response(url)

    monkeypatch.setattr(iris.data.requests, "get", get)

    result = iris.data.download([_url_nrt, _url_final], directory=tmp_path)

    assert len(result) == 2
    assert result[0] != result[1]
    assert result[0].name == result[1].name
    for file in result:
        assert file.read_bytes().decode() == (
            _url_nrt if file.parent.name == "nrt" else _url_final
        )


def test_decompress_directories(tmp_path: pathlib.Path):
    """Each archive is decompressed next to itself."""
    (archive,) = iris.data.download(
        urls=iris.data.urls_hek(
            time_stop=astropy.time.Time("2024-01-01"),
            obs_id=_obsid_b2,
            limit=1,
            sji=False,
        )
    )

    archives = []
    for name in ["a", "b"]:
        copy = tmp_path / name / archive.name
        copy.parent.mkdir()
        copy.write_bytes(archive.read_bytes())
        archives.append(copy)

    result = iris.data.decompress(archives)

    assert {file.relative_to(tmp_path).parts[0] for file in result} == {"a", "b"}
