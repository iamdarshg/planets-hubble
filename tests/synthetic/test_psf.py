import numpy as np

from synthetic.psf import EmpiricalPsfLibrary, PsfProvider


def test_empirical_psf_is_preferred_and_normalized() -> None:
    library = EmpiricalPsfLibrary()
    kernel = np.arange(25, dtype=np.float32).reshape(5, 5) + 1.0
    library.add(
        instrument="WFC3",
        detector="UVIS",
        filter_name="F606W",
        x=512.0,
        y=512.0,
        focus=0.1,
        kernel=kernel,
    )

    result = PsfProvider(library).render(
        instrument="WFC3",
        detector="UVIS",
        filter_name="F606W",
        x=510.0,
        y=514.0,
        wavelength_nm=600.0,
        focus=0.11,
    )

    np.testing.assert_allclose(result.kernel.sum(), 1.0, rtol=1e-6)
    assert result.metadata["tier"] == "empirical"
    assert result.metadata["matched_distance"] >= 0.0


def test_physics_fallback_has_wavelength_and_focus_dependent_structure() -> None:
    provider = PsfProvider()
    blue = provider.render(
        instrument="WFC3",
        detector="IR",
        filter_name="F160W",
        x=100.0,
        y=200.0,
        wavelength_nm=900.0,
        focus=0.0,
        jitter=(0.0, 0.0),
    )
    red = provider.render(
        instrument="WFC3",
        detector="IR",
        filter_name="F160W",
        x=100.0,
        y=200.0,
        wavelength_nm=1600.0,
        focus=0.4,
        jitter=(0.2, -0.1),
    )

    assert blue.metadata["tier"] == "physics_fallback"
    assert blue.kernel.shape == red.kernel.shape
    np.testing.assert_allclose(blue.kernel.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(red.kernel.sum(), 1.0, rtol=1e-6)
    assert not np.allclose(blue.kernel, red.kernel)
    assert blue.kernel[blue.kernel.shape[0] // 2, blue.kernel.shape[1] // 2] > 0.0
