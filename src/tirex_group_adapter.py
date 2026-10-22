"""Add simultaneous native-unit groups to the unchanged scalar adapter.

Only patch validation changes. The inherited forecast method already performs
one cloned tensor assignment at token 63, captures after that assignment, and
enforces exactly one hook execution with unconditional cleanup. A list index
selects an N-by-K block in that same assignment; no sequential forecasts or
successively patched donor states are involved.
"""

import numpy as np
import torch

from .tirex_intervention_adapter import UNIT_COUNT, TirexAdapter


class GroupTirexAdapter(TirexAdapter):
    """Accept {'layer': int, 'units': distinct list[int], 'values': array[N,K]}.

    Values in each row must be supplied together from its original donor by
    the caller. This adapter preserves row/column pairing, rejects broadcasting,
    and copies both indices and values before passing them to the frozen hook.
    No patch uses the inherited capture-only path unchanged.
    """

    def _validate_patch(self, patch, batch_size):
        if patch is None:
            return None
        if not isinstance(patch, dict) or set(patch) != {"layer", "units", "values"}:
            raise ValueError("group patch must contain exactly layer, units, values")
        layer, units = patch["layer"], patch["units"]
        if type(layer) is not int or layer not in self.layers:
            raise ValueError("group patch layer must be one of the selected native layers")
        if (not isinstance(units, list) or not units
                or any(type(unit) is not int or not 0 <= unit < UNIT_COUNT for unit in units)):
            raise ValueError("group units must be a nonempty list of integer indices in 0..1407")
        if len(set(units)) != len(units):
            raise ValueError("group unit indices must be unique")
        values = np.asarray(patch["values"], dtype=np.float32)
        if values.shape != (batch_size, len(units)) or not np.isfinite(values).all():
            raise ValueError("group values must have finite exact shape (batch_size, group_size)")
        # The frozen hook uses value[:, TOKEN_INDEX, patch['unit']] = values.
        # A Python list index yields precisely the required N-by-K slice.
        return {"layer": layer, "unit": units.copy(), "values": torch.from_numpy(values.copy())}
