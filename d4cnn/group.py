# pylint: disable=no-member, invalid-name, missing-docstring
import torch

d4_mul = [
    [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
    [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
    [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
    [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]
]
d4_inv = [0, 1, 2, 3, 4, 6, 5, 7]


def test_group():
    e = 0
    G = list(range(8))

    # inverse
    assert all(d4_mul[d4_inv[a]][a] == e for a in G)
    assert all(d4_mul[a][d4_inv[a]] == e for a in G)

    # associativity
    assert all(d4_mul[d4_mul[a][b]][c] == d4_mul[a][d4_mul[b][c]] for a in G for b in G for c in G)


@torch.jit.script
def image_action(u: int, image, h: int, w: int):
    if u == 0:
        return image
    if u == 1:
        return image.flip(w)
    if u == 2:
        return image.flip(h)
    if u == 3:
        return image.flip(w, h)
    if u == 4:
        return image.transpose(w, h)
    if u == 5:
        return image.transpose(w, h).flip(w)
    if u == 6:
        return image.transpose(w, h).flip(h)
    if u == 7:
        return image.transpose(w, h).flip(w, h)
    raise ValueError()


def field_action(u: int, field, g: int, h: int, w: int):
    field = image_action(u, field, h, w).contiguous()  # f'(xv) = f(u^-1 x u  v)
    i = field.new_tensor(d4_mul[d4_inv[u]], dtype=torch.long)
    return field.index_select(g, i)  # f'(xv) = f(u^-1 x u  u^-1 v)


def test_field_representation():
    f = torch.rand(8, 10, 10)

    for u in range(8):
        for v in range(8):
            vf = field_action(v, f, 0, 1, 2)
            uvf = field_action(u, vf, 0, 1, 2)
            uvf_ = field_action(d4_mul[u][v], f, 0, 1, 2)
            assert (uvf == uvf_).all()

    for u in range(8):
        uf = field_action(u, f, 0, 1, 2)
        iuf = field_action(d4_inv[u], uf, 0, 1, 2)
        assert (f == iuf).all()
