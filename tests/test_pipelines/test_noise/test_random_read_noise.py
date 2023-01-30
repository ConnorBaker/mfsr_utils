from hypothesis import given
from hypothesis import strategies as st

from mfsr_utils.pipelines.noise import random_read_noise


@given(shot_noise=st.floats(0.0, 1.0, exclude_min=True))
def test_random_read_noise_is_bounded(shot_noise: float) -> None:
    """
    Tests that random_read_noise is in the range [0, 1].

    Args:
        shot_noise: Shot noise
    """
    read_noise = random_read_noise(shot_noise)
    assert 0.0 <= read_noise <= 1.0
