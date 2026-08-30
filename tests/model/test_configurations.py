from model.configurations import research_config
from model.astromamba_h import AstroMambaH


def test_research_config_targets_the_specified_parameter_range() -> None:
    config = research_config()
    model = AstroMambaH(config)

    assert config.input_size == (720, 1280)
    assert config.canonical_wavelength_bins == 16
    # V2 deliberately targets a more useful 50-65M budget after removing
    # redundant fusion/temporal depth and reallocating capacity to FPN,
    # modality-set encoders, and the spatial decoder.
    assert 50_000_000 <= model.parameter_count <= 65_000_000
