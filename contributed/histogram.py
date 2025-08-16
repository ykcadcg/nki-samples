import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def histogram(values, counts, length):

    """
    Compute a histogram over integer bin IDs with HBM atomics.
    values : [N, 1] int32   (bin ids in [0, B))
    counts : [B, 1] int32   (zero-initialized; output)
    """

    assert values.dtype == nl.int32 and counts.dtype == nl.int32, "int32 required"
    assert length == values.shape[0], "length must equal N"
    T = int(nl.tile_size.pmax)
    assert length % T == 0, "require N multiple of tile_size.pmax"

    for off in nl.affine_range(0, length, step=T):
        bins_t = nl.load(values[off:off + T, :])                    # [T,1]
        ones   = nl.full_like(bins_t, 1, dtype=nl.int32)            # [T,1]
        nl.atomic_rmw(counts[bins_t, :], value=ones, op=np.add)     # add only
    return counts


def check_correct(B=64, TILES=32, seed=0) -> bool:
    rng = np.random.default_rng(seed)
    T = int(nl.tile_size.pmax)
    N = T * TILES
    values = rng.integers(0, B, size=(N, 1), dtype=np.int32)
    counts = np.zeros((B, 1), dtype=np.int32)

    run = nki.baremetal()(histogram)
    out = run(values, counts, N)

    ref = np.bincount(values.ravel(), minlength=B).astype(np.int64)
    got = out[:, 0].astype(np.int64)
    ok = bool(np.array_equal(got, ref))
    print(f"check_correct: N={N} B={B} T={T} -> match={ok}")
    return ok


def benchmark_kernel(B=256, TILES=256, iters=50, warmup=10, seed=1):
    rng = np.random.default_rng(seed)
    T = int(nl.tile_size.pmax)
    N = T * TILES
    values = rng.integers(0, B, size=(N, 1), dtype=np.int32)
    counts = np.zeros((B, 1), dtype=np.int32)

    @nki.benchmark(iters=iters, warmup=warmup)
    def _bench():
        counts.fill(0)
        return histogram(values, counts, N)

    stats = _bench()
    print(f"benchmark: N={N} B={B} T={T} iters={iters} warmup={warmup}")
    return stats


def main():
    assert check_correct(B=64, TILES=32)
    benchmark_kernel(B=256, TILES=128, iters=30, warmup=10)


if __name__ == "__main__":
    main()
