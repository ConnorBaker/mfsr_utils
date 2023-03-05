from hypothesis import HealthCheck, settings

settings.register_profile(
    "dev",
    deadline=None,
    max_examples=32,
    suppress_health_check=[HealthCheck.filter_too_much],
)
settings.register_profile(
    "dev_slow",
    deadline=None,
    max_examples=64,
    suppress_health_check=[HealthCheck.filter_too_much],
)
settings.register_profile(
    "ci", deadline=None, max_examples=256, suppress_health_check=[HealthCheck.filter_too_much]
)
settings.register_profile(
    "ci_slow",
    deadline=None,
    max_examples=512,
    suppress_health_check=[HealthCheck.filter_too_much],
)
