import pytest
import numpy as np
import astropy.time
from iris.sji import _dust


def test_urls():
    url_flat, url_badpix = _dust._urls()
    assert url_flat.startswith(_dust._url_sswdb)
    assert url_flat.endswith("_flat.genx")
    assert url_badpix.endswith("_badpix.geny")
    assert url_flat.removesuffix("_flat.genx") == url_badpix.removesuffix(
        "_badpix.geny"
    )


@pytest.mark.parametrize("window", ["SJI_1400", "SJI_2796"])
def test_ccd(window: str):
    result = _dust._ccd(window, astropy.time.Time("2019-09-30T18:00"))
    assert result.shape == _dust._shape_ccd
    assert result.dtype == bool
    assert 0 < np.sum(result) < 0.01 * result.size


def test_ccd_unknown_window():
    with pytest.raises(ValueError, match="no flat fields"):
        _dust._ccd("SJI_9999", astropy.time.Time("2019-09-30T18:00"))


@pytest.mark.parametrize(
    argnames="window,binning,expected",
    argvalues=[
        ("SJI_1400", (1, 1), 2),
        ("SJI_1400", (2, 2), 2),
        ("SJI_2796", (4, 4), 2),
        ("SJI_2832", (1, 1), 2),
        ("SJI_2832", (2, 2), 3),
        ("SJI_2832", (1, 4), 4),
        ("SJI_2832", (2, 4), 2),
    ],
)
def test_radius(window: str, binning: tuple[int, int], expected: int):
    assert _dust._radius(window, binning) == expected


def test_offsets_unknown_window():
    with pytest.raises(ValueError, match="Unrecognized"):
        _dust._offsets("SJI_9999", astropy.time.Time("2019-09-30T18:00"), (2, 2))


def test_pixels():
    offsets = np.array([[0.5, 0.0], [-1.25, 2.5]])
    slit = np.array([[10.0, 20.0], [11.0, 20.0]])

    t, x, y = _dust._pixels(offsets, slit)

    expected = {
        (0, 10, 20),
        (0, 11, 20),
        (0, 8, 22),
        (0, 9, 23),
        (1, 11, 20),
        (1, 12, 20),
        (1, 9, 22),
        (1, 10, 23),
    }
    assert set(zip(t.tolist(), x.tolist(), y.tolist())) == expected


@pytest.mark.parametrize("shift", [(0, 0), (2, -3), (-7, 7)])
def test_shift(shift: tuple[int, int]):
    """The shift which puts the dust on the dark pixels is found."""

    rng = np.random.default_rng(seed=0)

    num_t, num_y, num_x = 3, 60, 80
    offsets = rng.uniform(-15, 15, size=(40, 2))
    slit = np.array([[40.0, 30.0], [42.0, 30.0], [44.0, 30.0]])

    image = rng.uniform(10, 20, size=(num_t, num_y, num_x))
    t, x, y = _dust._pixels(offsets, slit + np.array(shift))
    image[t, y, x] = 0

    result = _dust._shift(
        offsets=offsets,
        slit=slit,
        image=image,
        crval=np.zeros((num_t, 2)),
        crpix=np.zeros(2),
        cdelt=np.array([0.33, 0.33]),
        pc=np.broadcast_to(np.identity(2), (num_t, 2, 2)),
    )

    assert result == shift


def test_shift_off_disk():
    """
    The pixels off the disk of the Sun are left out of the fit,
    so a dark sky does not move the dust.
    """

    num_t, num_y, num_x = 1, 40, 40
    offsets = np.array([[0.0, 0.0]])
    slit = np.array([[20.0, 20.0]])

    # Dark everywhere except the pixel the dust is on
    image = np.zeros((num_t, num_y, num_x))
    image[0, 20, 20] = 1

    result = _dust._shift(
        offsets=offsets,
        slit=slit,
        image=image,
        crval=np.array([[2000.0, 0.0]]),
        crpix=np.array([20.0, 20.0]),
        cdelt=np.array([0.33, 0.33]),
        pc=np.identity(2)[np.newaxis],
    )

    assert result == (0, 0)
