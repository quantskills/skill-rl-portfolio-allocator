import numpy as np

from scripts.action_transform import transform_delta_action


def test_delta_transform_clips_delta_and_finishes_with_unit_l1():
    previous = np.array([0.6, -0.4])
    out = transform_delta_action(
        np.array([100.0, -100.0]), previous, max_delta=0.2, ema_alpha=1.0
    )

    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.abs(out).sum(), 1.0, atol=1e-12)


def test_zero_delta_preserves_previous_exactly():
    previous = np.array([0.75, -0.25])

    out = transform_delta_action(np.zeros(2), previous, max_delta=0.2, ema_alpha=0.3)

    np.testing.assert_array_equal(out, previous)


def test_delta_transform_applies_ema_after_target_normalization():
    previous = np.array([1.0, 0.0])
    out = transform_delta_action(
        np.array([0.0, 1.0]), previous, max_delta=1.0, ema_alpha=0.5
    )

    target = np.array([1.0, np.tanh(1.0)])
    target /= np.abs(target).sum()
    expected = 0.5 * target + 0.5 * previous
    expected /= np.abs(expected).sum()
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_delta_transform_rejects_nonfinite_inputs():
    with np.testing.assert_raises(ValueError):
        transform_delta_action(np.array([np.nan]), np.array([1.0]), 0.2, 0.5)


def test_max_delta_zero_keeps_previous_and_positive_limit_clips_raw_tanh_delta():
    previous = np.array([0.75, -0.25])
    np.testing.assert_array_equal(
        transform_delta_action(np.array([100.0, -100.0]), previous, 0.0, 1.0),
        previous,
    )

    raw = np.array([100.0, -100.0])
    max_delta = 0.2
    target = previous + np.clip(np.tanh(raw), -max_delta, max_delta)
    target /= np.abs(target).sum()
    np.testing.assert_allclose(
        transform_delta_action(raw, previous, max_delta, 1.0), target, atol=1e-12
    )


def test_max_delta_must_be_finite_and_nonnegative():
    previous = np.array([1.0])
    for bad in (-1.0, np.nan, np.inf):
        with np.testing.assert_raises(ValueError):
            transform_delta_action(np.array([0.1]), previous, bad, 1.0)
