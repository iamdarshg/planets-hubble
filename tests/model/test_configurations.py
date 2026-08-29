from model.configurations import research_config
from model.astromamba_h import AstroMambaH


def test_research_config_targets_the_specified_parameter_range() -> None:
    config = research_config()
    model = AstroMambaH(config)

    assert config.input_size == (720, 1280)
    assert config.canonical_wavelength_bins == 16
    assert 70_000_000 <= model.parameter_count <= 90_000_000
