"""Implementation of matrix embedding.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.optimize


def efficiency(
    alpha: float
) -> float:
    """Approximates embedding efficiency for Hamming code.

    Assumes last block to be padded.

    :param alpha: embedding rate in bits per element
    :type alpha: float
    :return: embedding efficiency in bits per change
    :rtype: float

    :Example:

    >>> e = cl.coding.efficiency(0.4)  # e at alpha=0.4
    """
    assert alpha >= 1e-3, 'instable for very low embedding rates'

    def mse_p(p) -> float:
        """SS error from blocksize."""
        return (p / (2**p - 1) - alpha)**2

    p = scipy.optimize.fminbound(mse_p, 1, 100, xtol=1e-4)
    return p / (1 - 2.**-p)

def hamming_params(k: int) -> tuple[int, float, float, float]:
    """Return parameters for a Hamming code with message length k.

    :param k: Number of message bits.
    :return: Tuple ``(n, alpha, rc, e)`` where:
        - n: block size (codeword length).
        - alpha: embedding rate in bits per element.
        - rc: relative change rate (changes per element).
        - e: embedding efficiency in bits per change.
    """
    n = 2**k - 1
    alpha = k / n
    rc = (2**k - 1) / (2**k)
    e = k / rc
    return n, alpha, rc, e

def generate_parity_matrix(k: int) -> np.ndarray:
    """Generate the Hamming parity-check matrix for message length k.

    The matrix columns are the binary representation of 1..n, with
    n = 2**k - 1.
    example for k=3:
H = [[0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [1, 0, 1, 0, 1, 0, 1]]

    example for k=4:
H = [[0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
    [0 0 0 1 1 1 1 0 0 0 0 1 1 1 1]
    [0 1 1 0 0 1 1 0 0 1 1 0 0 1 1]
    [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]]

    :param k: Number of message bits.
    :return: Parity-check matrix H of shape (k, n).
    """
    n = 2**k - 1
    H = np.zeros((k, n), dtype=int)
    for i in range(1, n + 1):
        bin_str = bin(i)[2:].zfill(k)
        H[:, i - 1] = [int(b) for b in bin_str]
    return H

def embed_lsbm_hamming(
    x: np.ndarray,
    m: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Embed message bits into a block using LSBM with a Hamming code.

    This computes the syndrome of the current LSBs, determines the required
    single-position change, and applies an LSB matching step at that position.
    Only one element is modified per block when a change is needed.

    :param x: Cover block of shape (n,) where n = 2**k - 1.
    :param m: Message bits 
    :param k: Number of message bits.
    :param rng: Optional RNG for tie-breaking between +1 and -1.
    :return: Stego block of shape (n,).
    """
    
    n = 2**k - 1
    assert len(x) == n and len(m) == k

    H = generate_parity_matrix(k)
    y = x.astype(np.int16, copy=True) if np.issubdtype(x.dtype, np.integer) else x.copy()
    
    # The change vector is c = m - H * y mod 2, 
    # which indicates the single bit flip needed to correct the syndrome.
    syndrome = np.dot(H, y % 2) % 2 
    diff = (m - syndrome) % 2

    x_min, x_max = 0, 255

    if np.any(diff):
        diff_val = int(diff.dot(1 << np.arange(diff.shape[0])[::-1]))
        if diff_val > 0:
            idx = diff_val - 1
            value = int(y[idx])
            if value <= x_min:
                y[idx] = value + 1
            elif value >= x_max:
                y[idx] = value - 1
            else:
                step = rng.choice([-1, 1]) if rng is not None else np.random.choice([-1, 1]) # here we need a random gerator to select if we go up or down
                y[idx] = value + step

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = np.clip(y, info.min, info.max).astype(x.dtype, copy=False)
    return y


def embed_message_lsbm_hamming(
    cover: np.ndarray,
    msg: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Embed an arbitrary-length message using LSBM with a Hamming code.
    
    The message elements should be in [0, 255] (bytes). They are unpacked into bits.
    
    :param cover: Cover array of arbitrary shape.
    :param msg: Message array with values from 0 to 255.
    :param k: Number of message bits per block.
    :param rng: Optional RNG for tie-breaking.
    :return: Stego array of the same shape as the cover with the message embedded.
    """
    msg_bits = np.unpackbits(np.asarray(msg, dtype=np.uint8))
    
    n = 2**k - 1
    num_blocks = int(np.ceil(len(msg_bits) / k))
    required_len = num_blocks * n
    
    if cover.size < required_len:
        raise ValueError('Cover is too short to embed the message.')
        
    # Pad message bits with zeros if length is not a multiple of k
    if len(msg_bits) % k != 0:
        pad_len = k - (len(msg_bits) % k)
        msg_bits = np.pad(msg_bits, (0, pad_len), constant_values=0)
        
    stego = cover.copy()
    stego_flat = stego.reshape(-1)
    
    msg_blocks = msg_bits.reshape(-1, k)
    
    for i in range(num_blocks):
        start = i * n
        end = start + n
        stego_flat[start:end] = embed_lsbm_hamming(
            stego_flat[start:end],
            msg_blocks[i],
            k,
            rng=rng
        )
        
    return stego


def extract_lsbm_hamming(
    y: np.ndarray,
    k: int,
    num_bits: int | None = None
) -> np.ndarray:
    """Decode message bits from stego data using the Hamming parity check.

    The input can be a single block of length ``n = 2**k - 1`` or any
    array-like. For multi-dimensional inputs, the data is flattened in
    row-major order into blocks of length ``n``. Any trailing elements that do
    not form a complete block are ignored. Use ``num_bits`` to truncate the
    output to a known message length.

    :param y: Stego data containing embedded LSBs.
    :param k: Number of message bits per block.
    :param num_bits: Optional total number of message bits to return.
    :return: Decoded message bits as a 1D array of length ``k * num_blocks``.
    """
    n = 2**k - 1
    H = generate_parity_matrix(k)

    y_arr = np.asarray(y)
    if y_arr.ndim == 1 and y_arr.shape[0] == n:
        msg = np.dot(H, y_arr % 2) % 2
    else:
        y_flat = y_arr.reshape(-1)
        num_blocks = y_flat.size // n
        if num_blocks <= 0:
            raise ValueError('not enough elements for a single block')
        y_blocks = y_flat[:num_blocks * n].reshape(num_blocks, n)
        msg = (H @ (y_blocks % 2).T) % 2
        msg = msg.T.reshape(-1)

    if num_bits is not None:
        if num_bits < 0:
            raise ValueError('num_bits must be non-negative')
        msg = msg[:num_bits]

    return msg.astype(np.uint8, copy=False)


def simulate_embedding(
    k: int,
    num_blocks: int,
    rng: np.random.Generator | None = None
) -> int:
    """Simulate the number of changed blocks for Hamming embedding.

    Each block changes with probability (2**k - 1) / (2**k). The result is
    sampled from a binomial distribution.

    :param k: Number of message bits.
    :param num_blocks: Number of independent blocks to simulate.
    :param rng: Optional RNG for reproducibility.
    :return: Number of changed blocks.
    """
    prob_change = (2**k - 1) / (2**k)
    if rng is None:
        return int(np.random.binomial(num_blocks, prob_change))
    return int(rng.binomial(num_blocks, prob_change))

def _mix_for_alpha(alpha: float) -> tuple[int, int, float, float]:
    """Find the two neighboring Hamming codes that bracket alpha.

    This returns k and k+1 along with linear weights for interpolation between
    their embedding rates.

    :param alpha: Target embedding rate in (0, 1].
    :return: Tuple (k, k+1, weight_k, weight_k1).
    """
    if not (0 < alpha <= 1):
        raise ValueError('alpha must be in (0, 1]')

    k = 1
    while True:
        _, alpha_k, _, _ = hamming_params(k)
        _, alpha_k1, _, _ = hamming_params(k + 1)
        if alpha_k1 <= alpha <= alpha_k:
            weight_k = (alpha - alpha_k1) / (alpha_k - alpha_k1)
            return k, k + 1, weight_k, 1.0 - weight_k
        k += 1

def _changes_per_element(alpha: float) -> float:
    """Compute expected changes per element for a target alpha.

    Uses linear interpolation between neighboring Hamming codes when alpha
    is not exactly achievable by a single code.

    :param alpha: Target embedding rate in (0, 1].
    :return: Expected changes per element.
    """
    k, k1, weight_k, weight_k1 = _mix_for_alpha(alpha)
    n_k, _, rc_k, _ = hamming_params(k)
    n_k1, _, rc_k1, _ = hamming_params(k1)
    return weight_k * (rc_k / n_k) + weight_k1 * (rc_k1 / n_k1)

def analytical_e(alpha: float) -> float:
    """Calculate embedding efficiency e for a given alpha.

    :param alpha: Target embedding rate in (0, 1].
    :return: Embedding efficiency in bits per change.
    """
    changes_per_element = _changes_per_element(alpha)
    return alpha / changes_per_element

def simulate_embedding_alpha(
    alpha: float,
    num_elements: int,
    rng: np.random.Generator | None = None
) -> int:
    """Simulate the number of changes for a target embedding rate alpha.

    This uses the interpolated change probability per element and samples
    from a binomial distribution.

    :param alpha: Target embedding rate in (0, 1].
    :param num_elements: Number of elements to simulate.
    :param rng: Optional RNG for reproducibility.
    :return: Number of changed elements.
    """
    changes_per_element = _changes_per_element(alpha)
    if rng is None:
        return int(np.random.binomial(num_elements, changes_per_element))
    return int(rng.binomial(num_elements, changes_per_element))
