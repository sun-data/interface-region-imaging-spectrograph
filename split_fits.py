import pathlib
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits


def split_cubes(
    path_source: list[str | pathlib.Path],
    directory_destination: str | pathlib.Path,
) -> list[str | pathlib.Path]:
    """
    Split fits files containing wavelength, x, and y axes into separate fits files
    containing only wavelength and y axes.

    Parameters
    ----------
    path_source
        A list of paths to source files to be converted
    directory_destination
        A directory in which to place the results.
    """

    directory_destination = pathlib.Path(directory_destination)
    directory_destination.mkdir(exist_ok=True, parents=True)

    for path in path_source:

        path = pathlib.Path(path)

        hdul = astropy.io.fits.open(path)

        index_main = 0
        index_aux1 = ~1
        index_aux2 = ~2
        index_image = slice(index_main + 1, index_aux1)

        hdu_main = hdul[index_main]
        hdu_image = hdul[index_image]
        hdu_aux1 = hdul[index_aux1]
        hdu_aux2 = hdul[index_aux2]

        num_steps = hdu_main.header["NEXP"]

        date_obs = astropy.time.Time(hdu_main.header["DATE_OBS"])
        time = date_obs + hdu_aux1.data[..., hdu_aux1.header["TIME"]] * u.s

        for step in range(num_steps):

            slice_step = slice(step, step + 1)

            hdu_step_main = hdu_main.copy()
            hdu_step_main.header["DATE_OBS"] = time[step].fits

            hdu_step_image = [hdu.copy() for hdu in hdu_image]
            for hdu in hdu_step_image:
                hdu.header.update(astropy.wcs.WCS(hdu.header)[slice_step].to_header())
                hdu.data = hdu.data[slice_step]

            hdu_step_aux1 = hdu_aux1.copy()
            hdu_step_aux1.data = hdu_step_aux1.data[slice_step]

            hdu_step_aux2 = hdu_aux2.copy()

            hdul_step = (
                [hdu_step_main] + hdu_step_image + [hdu_step_aux1, hdu_step_aux2]
            )
            hdul_step = astropy.io.fits.HDUList(hdul_step)

            filename = directory_destination / f"{path.stem}_s{step}{path.suffix}"

            hdul_step.writeto(filename, overwrite=True)


directory_source = pathlib.Path(r"C:\Users\byrdie\Kankelborg-Group\interface-region-imaging-spectrograph\iris_l2_20240222_214410_3680334922_raster")
path_source = list(directory_source.glob("*"))

split_cubes(
    path_source=path_source,
    directory_destination=r"C:\Users\byrdie\Kankelborg-Group\interface-region-imaging-spectrograph\iris_l2_20240222_214410_3680334922_raster_split"
)
