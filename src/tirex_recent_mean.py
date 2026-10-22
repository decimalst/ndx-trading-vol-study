"""Calibration-only one-feature surrogate for the fixed L0/U666 activation."""
import numpy as np


def _arrays(histories, activations):
    x, a = np.asarray(histories, dtype=float), np.asarray(activations, dtype=float)
    if (x.ndim != 3 or x.shape[0] < 2 or x.shape[1] != 2 or x.shape[2] < 32
            or a.shape != x.shape[:2] or not np.isfinite(x).all()
            or not np.isfinite(a).all()):
        raise ValueError('finite histories N>=2 x2xL>=32 and activations Nx2 required')
    return x[:, :, -32:].mean(axis=-1), a


def fit_readout(histories, activations):
    feature, a = _arrays(histories, activations)
    design = np.column_stack((np.ones(feature.size), feature.ravel()))
    coefficients, _, rank, _ = np.linalg.lstsq(design, a.ravel(), rcond=None)
    return {'coefficients': coefficients.tolist(), 'rank': int(rank),
            'calibration_pairs': len(feature), 'window': 32}


def _r2(actual, prediction):
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return None if denominator == 0 else float(1 - np.sum((actual - prediction) ** 2) / denominator)


def evaluate_readout(model, histories, activations):
    feature, actual = _arrays(histories, activations)
    coefficients = np.asarray(model['coefficients'], dtype=float)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all() or model['window'] != 32:
        raise ValueError('invalid frozen one-feature readout')
    prediction = coefficients[0] + coefficients[1] * feature
    actual_difference = actual[:, 1] - actual[:, 0]
    predicted_difference = prediction[:, 1] - prediction[:, 0]
    return {'n_pairs': len(actual), 'level_r2': _r2(actual, prediction),
            'paired_difference_r2': _r2(actual_difference, predicted_difference),
            'level_rmse': float(np.sqrt(np.mean((actual - prediction) ** 2))),
            'paired_difference_rmse': float(np.sqrt(np.mean((actual_difference - predicted_difference) ** 2)))}
