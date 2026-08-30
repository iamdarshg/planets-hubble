from model.configurations import research_config
from model.astromamba_h import AstroMambaH


def test_research_config_targets_the_specified_parameter_range() -> None:
    config = research_config()
    model = AstroMambaH(config)

    assert config.input_size == (720, 1280)
    assert config.canonical_wavelength_bins == 16
    # The research preset must land in the specification's 82-86M target
    # window by reallocating capacity across the existing model components.
    assert 82_000_000 <= model.parameter_count <= 86_000_000
