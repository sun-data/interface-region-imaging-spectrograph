import pathlib
import numpy as np
import astropy.wcs
import astropy.io.fits


def convert_raster_to_sitandstare(
    path_source: list[str | pathlib.Path],
    directory_destination: str | pathlib.Path,
) -> list[str | pathlib.Path]:
    """
    Convert FITS files generated from IRIS rasters (x, y, wavelength)
    to the same format as FITS files generated from IRIS sit-and-stare
    observations (time, y, wavelength).

    Parameters
    ----------
    path_source
        A list of paths to source files to be converted
    directory_destination
        A directory in which to place the results.
    """

    directory_destination = pathlib.Path(directory_destination)
    directory_destination.mkdir(exist_ok=True, parents=True)

    index_main = 0
    index_aux1 = ~1
    index_aux2 = ~2
    index_image = slice(index_main + 1, index_aux1)

    path_0 = path_source[0]
    hdul_prototype = astropy.io.fits.open(path_0)

    num_x = hdul_prototype[0].header["NRASTERP"]
    num_window = len(hdul_prototype[index_image])

    for i in range(num_x):

        hdul = hdul_prototype.copy()

        data_aux1 = []
        crval_x = []
        crval_y = []

        for w in range(num_window):

            index_window = w + 1

            hdu = hdul[index_window].copy()

            wcs = astropy.wcs.WCS(hdu.header)
            wcs = wcs[i : i + 1]
            wcs.wcs.cdelt[~0] = 0
            hdu.header.update(wcs.to_header())

            data = []
            data_aux1_w = []
            crval_x_w = []
            crval_y_w = []

            for path in path_source:
                hdul_t = astropy.io.fits.open(path)
                hdu_t = hdul_t[index_window]
                wcs_t = astropy.wcs.WCS(hdu_t.header)
                crval_x_w.append(wcs_t.wcs.crval[~0])
                crval_y_w.append(wcs_t.wcs.crval[~1])
                data.append(hdu_t.data[i])
                data_aux1_w.append(hdul_t[index_aux1].data[i])

            hdu.data = np.stack(data)
            data_aux1.append(np.stack(data_aux1_w))
            crval_x.append(np.stack(crval_x_w))
            crval_y.append(np.stack(crval_y_w))

            hdul[index_window] = hdu

        hdu_aux1 = hdul[index_aux1].copy()
        hdu_aux1.data = np.stack(data_aux1)
        hdu_aux1.data[..., hdu_aux1.header["XCENIX"]] = np.stack(crval_x).mean(0)
        hdu_aux1.data[..., hdu_aux1.header["YCENIX"]] = np.stack(crval_y).mean(0)
        hdul[index_aux1] = hdu_aux1

        hdul.writeto(
            directory_destination / f"{path_0.stem}_s{i}{path_0.suffix}",
            overwrite=True,
        )


directory_source = pathlib.Path(
    r"C:\Users\byrdie\Kankelborg-Group\interface-region-imaging-spectrograph\iris_l2_20240222_214410_3680334922_raster"
)
path_source = sorted(list(directory_source.glob("*")))

convert_raster_to_sitandstare(
    path_source=path_source,
    directory_destination=r"C:\Users\byrdie\Kankelborg-Group\interface-region-imaging-spectrograph\iris_l2_20240222_214410_3680334922_sitandstare",
)
