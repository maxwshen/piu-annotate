import numpy as np
import numpy.typing as npt


def calc_max_event_frequency(times: list[float], t: float) -> float:
    """ Given `times`: list of times where event happens (such as downpress),
        with units in seconds, and window size `t` (in seconds), 
        computes the highest event frequency in any sliding window `t`.
        Returns float: units = events per second
    """
    if len(times) == 0:
        return 0.0
    
    n = len(times)
    if n == 1:
        return 1.0 / t
    
    # Use two-pointer technique to maintain a sliding window
    left = 0
    max_frequency = 0.0
    
    for right in range(n):
        # Shrink window from left while it exceeds size t
        while left < right and times[right] - times[left] > t:
            left += 1
            
        # Calculate frequency in current window
        events_in_window = right - left + 1
        frequency = events_in_window / t
        max_frequency = max(max_frequency, frequency)

    return max_frequency


def smallest_positive_difference(
    queries: npt.NDArray, 
    refs: npt.NDArray, 
    shift: bool = False
):
    """
    Find the smallest positive difference between each query and the largest 
    reference value less than the query.
    
    Parameters:
    -----------
    queries : numpy.ndarray
        Input array of shape (n,)
    refs : numpy.ndarray
        Reference array of shape (m,)
    shift : bool
        If True, shift found indices left by 1 more. This is used to handle
        staggered brackets.
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (n,) with smallest positive differences
    """
    # Sort the reference array
    sorted_refs = np.sort(refs)
    
    # Find the indices of the largest values less than each query
    # Using searchsorted for efficient lookup
    indices = np.searchsorted(sorted_refs, queries, side='left') - 1

    if shift:
        indices -= 1

    # Handle cases where no reference is less than the query
    valid_mask = indices >= 0
    
    # Initialize result array
    result = np.full_like(queries, np.nan, dtype=float)
    
    # Compute the differences for valid indices
    result[valid_mask] = queries[valid_mask] - sorted_refs[indices[valid_mask]]
    return result


def find_longest_true_run(values: list[bool]) -> tuple[int, int]:
    """
    Find the start and end indices of the longest consecutive run of True values.
    
    Args:
        values (list[bool]): List of boolean values
        
    Returns:
        tuple[int, int]: (start_index, end_index) of the longest run.
                        If no True values exist, returns (-1, -1).
                        The end_index is inclusive.
    """
    if len(values) == 0:
        return (-1, -1)
    
    max_length = 0
    max_start = -1
    max_end = -1
    
    current_start = 0
    current_length = 0
    
    for i, value in enumerate(values):
        if value:
            # Extend current run
            if current_length == 0:
                current_start = i
            current_length += 1
            
            # Update max if current run is longer
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
                max_end = i
        else:
            # Reset current run
            current_length = 0
    
    return (max_start, max_end)


def extract_consecutive_true_runs(bools: list[bool]) -> list[tuple[int, int]]:
    """ Given a list of bools, returns a list of tuples of start_idx, end_idx
        (inclusive) for each consecutive run of True in `bools`.
    """
    if not bools:
        return []
    
    runs = []
    current_run_start = None
    
    for idx, val in enumerate(bools):
        if val and current_run_start is None:
            # Start of a new True run
            current_run_start = idx
        elif not val and current_run_start is not None:
            # End of a True run
            runs.append((current_run_start, idx - 1))
            current_run_start = None
    
    # Check if the last run goes to the end of the list
    if current_run_start is not None:
        runs.append((current_run_start, len(bools) - 1))
    
    return runs