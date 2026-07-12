"""_pad_grid_to_2x2 sits under EVERY BRACS embedding -- frozen baseline and LoRA alike. A bug
here does not show up as a crash; it shows up as a wrong number in RESULTS.md."""
import numpy as np
import torch

from extract_bracs_features import TILE, _pad_grid_to_2x2, roi_to_patches


def _feats(n, dim=4):
    return torch.ones(n, dim)


def _grid_ok(coords):
    return len({x for x, _ in coords}) >= 2 and len({y for _, y in coords}) >= 2


def test_single_tile_roi_is_padded_to_a_full_2x2():
    """~1/3 of BRACS ROIs are a single 512px tile. TITAN's get_alibi() raises IndexError on a
    1xN grid, so this pad is what makes those ROIs encodable at all."""
    f, c = _pad_grid_to_2x2(_feats(1), [(0, 0)])
    assert f.shape[0] == 4 and len(c) == 4
    assert _grid_ok(c)
    assert int((f.abs().sum(dim=1) == 0).sum()) == 3      # exactly 3 zero pads injected


def test_horizontal_strip_is_padded():
    f, c = _pad_grid_to_2x2(_feats(2), [(0, 0), (TILE, 0)])
    assert _grid_ok(c)
    assert int((f.abs().sum(dim=1) == 0).sum()) == f.shape[0] - 2


def test_vertical_strip_is_padded():
    f, c = _pad_grid_to_2x2(_feats(2), [(0, 0), (0, TILE)])
    assert _grid_ok(c)
    assert int((f.abs().sum(dim=1) == 0).sum()) == f.shape[0] - 2


def test_an_already_2x2_grid_is_left_completely_alone():
    """Real tissue patches must never be touched -- no reordering, no padding, no copies."""
    coords = [(0, 0), (TILE, 0), (0, TILE), (TILE, TILE)]
    f_in = _feats(4)
    f, c = _pad_grid_to_2x2(f_in, coords)
    assert f.shape[0] == 4
    assert torch.equal(f, f_in)
    assert [tuple(x) for x in c] == coords


def test_pads_are_exactly_zero_so_titan_masks_them_as_background():
    """The pad only works because TITAN's preprocess_features drops rows via `any(feature != 0)`.
    A pad that is not bitwise zero would be fed to attention as if it were tissue."""
    f, _ = _pad_grid_to_2x2(_feats(1), [(0, 0)])
    pads = f[1:]
    assert torch.count_nonzero(pads) == 0


def test_real_patches_keep_their_position_and_values():
    f_in = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 1.0   # nonzero
    f, c = _pad_grid_to_2x2(f_in, [(0, 0), (TILE, 0)])
    assert torch.equal(f[:2], f_in)
    assert [tuple(x) for x in c[:2]] == [(0, 0), (TILE, 0)]


# ------------------------------------------------------------ tiling

def test_a_fully_blank_roi_still_yields_one_tile():
    """Documented behaviour: 'an all-blank ROI keeps one centered tile so it still yields an
    embedding'. Without it the ROI silently vanishes from the manifest-aligned feature matrix."""
    from PIL import Image
    white = Image.new("RGB", (1024, 1024), (255, 255, 255))
    tiles, coords = roi_to_patches(white)
    assert len(tiles) == 1 and coords == [(0, 0)]


def test_tissue_tiles_are_kept_and_padded_to_full_tile_size():
    from PIL import Image
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 120, size=(1400, 1400, 3), dtype=np.uint8))
    tiles, coords = roi_to_patches(noisy)          # 1400 -> 700 after the 2x downsample
    assert len(tiles) == len(coords) >= 4          # 700px / 512 -> a 2x2 grid with edge tiles
    for t in tiles:
        assert t.size == (TILE, TILE)              # edge tiles white-padded, never upscaled
