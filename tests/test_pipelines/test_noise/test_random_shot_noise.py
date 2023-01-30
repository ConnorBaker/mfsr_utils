from hypothesis import assume, given
from hypothesis import strategies as st

from mfsr_utils.pipelines.noise import random_shot_noise


@given(
    min_shot_noise=st.floats(0.0, 1.0, exclude_min=True),
    max_shot_noise=st.floats(0.0, 1.0, exclude_min=True),
)
def test_random_shot_noise_is_bounded(min_shot_noise: float, max_shot_noise: float) -> None:
    """
    Tests that random_shot_noise is in the range [min_shot_noise, max_shot_noise].

    Args:
        min_shot_noise: Minimum shot noise
        max_shot_noise: Maximum shot noise
    """
    assume(min_shot_noise <= max_shot_noise)
    shot_noise = random_shot_noise(min_shot_noise, max_shot_noise)
    assert min_shot_noise <= shot_noise <= max_shot_noise
