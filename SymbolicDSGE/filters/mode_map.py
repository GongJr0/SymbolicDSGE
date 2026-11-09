from .base import Filter
from .gaussian import LinearKalman, ExtendedKalman, UnscentedKalman
from .resolvers import FilterMode


def get_filter(mode: str | FilterMode) -> type[Filter]:
    """Get the filter class corresponding to the given mode.

    Parameters
    ----------
    mode : str | FilterMode
        Filter mode. One of 'linear', 'extended', or 'unscented'.

    Returns
    -------
    type[Filter]
        Filter class corresponding to the given mode.

    Raises
    ------
    ValueError
        If the mode is not recognized.
    """
    if isinstance(mode, str):
        # Raises with well-worded error message
        mode = FilterMode(mode)

    match mode:
        case FilterMode.LINEAR:
            return LinearKalman
        case FilterMode.EXTENDED:
            return ExtendedKalman
        case FilterMode.UNSCENTED:
            return UnscentedKalman
