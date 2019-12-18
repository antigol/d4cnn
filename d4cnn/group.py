# pylint: disable=no-member, invalid-name, missing-docstring
import torch

d4_mul = [
    [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
    [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
    [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
    [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]
]
d4_inv = [0, 1, 2, 3, 4, 6, 5, 7]


def image_action(u, image, h, w):
    return image_all_actions(image, h, w)[u]


def field_action(u, field, g, h, w):
    return field_all_actions(field, g, h, w)[u]


def image_all_actions(image, h, w):
    e = image
    m1 = image.flip(w)
    m2 = image.flip(h)
    i = m1.flip(h)
    tr = image.transpose(w, h)
    m1tr = tr.flip(w)
    m2tr = tr.flip(h)
    itr = m1tr.flip(h)
    return [e, m1, m2, i, tr, m1tr, m2tr, itr]


def field_all_actions(field, g, h, w):
    # return [(xv) -> f(u^-1 x u  u^-1 v) for u in G]
    return [
        # field = (xv) -> f(u^-1 x u  v)
        field.contiguous().index_select(g, field.new_tensor(d4_mul[d4_inv[u]], dtype=torch.long))
        for u, field in enumerate(image_all_actions(field, h, w))
    ]


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
