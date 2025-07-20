from PIL import Image
import numpy as np
import os, math
"""
    'BOOL': ('1', 1),
    'GREYB': ('L', 8),
    'GREYI': ('I', 32),
    'GREYF': ('F', 32),
    'MAP': ('P', 8),
    'RGB': ('RGB', 8 * 3),
    'RGBA': ('RGBA', 8 * 4), # only .png
    'PRINT': ('CMYK', 8 * 4),# green, red, yellow, black
    'BRIGHT': ('YCbCr', 8 * 3)
"""
class IMGStruct:
    def __init__(self, img: Image, arr=None):
        self.img = img
        img.load()
        self.arr = np.asarray(img) if arr is None else arr
    def __enter__(self): return self
    def __exit__(self, tp, arg, evt):
        self.img.close()
        self.img = self.arr = None

    def close(self): self.__exit__(None, '', 0)

    @staticmethod
    def mapRGBgreyb(img, rgb):
        new = IMGStruct(Image.new('L', img.img.size), mapRGBgreyb(img.arr, rgb))
        new.arr.resize(new.arr.size)
        new.img.putdata(new.arr)
        new.arr.resize(new.img.size)
        return new

    @staticmethod
    def mapGREYBbin(img, mid: int):
        new = IMGStruct(Image.new('1', img.img.size), mapGREYBbin(img.arr, mid))
        new.arr.resize(new.arr.size)
        new.img.putdata(new.arr)
        new.arr.resize(new.img.size)
        return new

    def save(self, path, *args, **kwargs):
        size = self.arr.shape
        self.arr.resize(self.arr.size)
        self.img.putdata(self.arr)
        self.arr.resize(size)
        self.img.save(path, *args, **kwargs)


def mapRGBgreyb(arr: np.array, rgb=(1, 1, 1)) -> np.uint8:
    return np.uint8( np.around( np.median( arr * rgb, axis=2 ) ) )

def mapGREYBbin(arr: np.array, mid: int):
    resver_size = arr.shape[1], arr.shape[0] // 8
    arr = np.uint8( arr > mid ).T.reshape( (resver_size[0] * resver_size[1], 8) )
    for i in range(1, 8):
        arr[:, i] >>= i
    return arr.sum(axis=1, dtype=np.uint8).reshape( resver_size ).T

'''
:condition
0 <= a < b <= 255
0 <= q <= 1
0 <= p, c1, c2 <= 255
:principle
c1 = p*q + a(1 - q)
c2 = p*q + b(1 - q)
:solve
q = 1 - (c2 - c1) / (b - a)
p = (b * c1 - a * c2) / [(b - a) - (c2 - c1)]
:depart
c1 <= c2
c2 - c1 <= b - a
b * c1 >= a * c2
:system
c1 = H1[c1'] = k * c1' + e1
c2 = H2[c2'] = k * c2' + e2
:inequality limit
&a = min{c2' - c1'} = (e1 - e2) / k
&b = max{c2' - c1'} = (b - a) / k + &a
&f = min{b * c1 - a * c2} = (a * e2 - b * e1) / k
:solve equation
k = (b - a) / (&b - &a)
e1 = k * (&f + a * &a) / (a - b)
e2 = k * (&f + b * &a) / (a - b)
:judgment
k ~= 0.5
e1 ~= 0.
e2 ~= 127.5
:warn

'''
def phantom_tank(new: IMGStruct, img0: IMGStruct, img1: IMGStruct, bk0=0, bk1=255):
    """:TODO: Misk two grey image into one RGBA photo, the color can change because backgroung translate(in grey).
    new: RGBA format Image IMGStruct. It's data did not cause any different to the result
    img0: L format Image IMGStruct. Image want to show when background with a color of bk0
    img1: L format Image IMGStruct. Image want to show when backgroung with a color of bk1
    bk0: Integer. defalut with 0
    bk1: Integer. defalut with 255
    :NOTES:
    * backgroung only have one simple color and significant. 0 <= bk0 << bk1 <= 255
    * All images parameters have the same photo size. if necessery, using function paste_image()
       transform images into a same size
    * Only support show out grey color, split the RGB photo to select a best alpha as input
    """
    assert bk0 < bk1, "If what to get such a different background, Please swap image 0 and image 1"
    assert img0.img.size == img1.img.size == new.img.size
    arr0 = np.int16(img0.arr)
    arr1 = np.int16(img1.arr)
    # arrdx = arr1 - arr0
    # c_min = arrdx.min()     # (e1 - e2) / k
    # c_max = arrdx.max()     # (b - a) / k + c_min
    # # d_min = (bk1 * arr0 - bk0 * arr1).min()     # (a * e2 - b * e1) / k
    # k = (bk1 - bk0) / (c_max - c_min)
    # # e1 = k * (d_min + bk0 * c_min) / (bk0 - bk1)
    # # e2 = k * (d_min + bk1 * c_min) / (bk0 - bk1)
    # de = - k * c_min
    # dx = (k * arrdx + de)#; np.clip(dx, 0, (bk1 - bk0), out=dx)
    # q = 1 - dx / (bk1 - bk0); q*=255; np.clip(q,0,255,out=q);np.around(q,out=q); q=q.astype(np.uint8)
    # e1 = - k * arr0.min()
    # e2 = e1 + de
    # arr0 = arr0 * k + e1
    # arr1 = arr1 * k + e2
    # print(c_min, c_max, )#d_min)  #-142 254 -7620
    # print(k, e1, e2)
    # # print(arr0)
    # # print(arr1)
    # w = (bk1 - bk0) - dx
    # # np.clip(w, 0, None, out=w)
    # # q = np.around(w * (255 / (bk1 - bk0))); np.clip(q,0,255,out=q)
    # # print(q, (q > 255).any(), (q < 0).any())
    # "never used np.put !!!!"
    # # q = q.astype(np.uint8, copy=False)#np.uint8(q)
    # zero = w == 0
    # w[zero] = 1
    # p = np.around((bk1 * arr0 - bk0 * arr1) / w)
    # p = np.clip(p, 0, 255, out=p).astype(np.uint8)
    # p[zero] = 255
    p, q = _phantom_base(arr0, arr1, bk0, bk1)

    a = Image.fromarray(p, mode='L')
    b = Image.fromarray(q, mode='L')
    try:
        "never used new.img.paste(a, mask=b)   !!!!"
        new.close()
        new.img = Image.merge('RGBA', (a, a, a, b))
    finally:
        a.close()
        b.close()
    return new.img


_between = lambda x, ab: ab[0]<x<ab[1]

def optimiz(A, B, C, D, E, F, G=0, H=0):
    A -= G; B -= G; C -= H; D -= H
    if A <= 0 <= B:
        if C <= 0 <= D:
            if E <= 0 <= F:
                return G, H
    lines = [
        (max(C, A + E), min(D, A + F)),  # e1 = A
        (max(C, B + E), min(D, B + F)),  # e1 = B
        (max(A, D - F), min(B, D - E)),  # e2 = D
        (max(A, C - F), min(B, C - E)),  # e2 = C
        (max(A, C - F), min(B, D - F)),  # e2 = e1 + F
        (max(C - E, A), min(B, D - E))  # e2 = e1 + E
    ]
    points = [
        (A, D) if F + A >= D else None,
        (A, F + A) if D > F + A > C else None,
        (A, E + A) if E + A > C else None,
        (A, C) if E + A <= C <= F + A else None,
        (B, D) if F + B >= D >= E + B else None,
        (B, F + B) if D > F + B else None,
        (B, E + B) if D > E + B > C else None,
        (B, C) if E + B <= C else None,
        (B, D) if D - E >= B >= D - F else None,
        (D - E, D) if B > D - E else None,
        (D - F, D) if B > D - F > A else None,
        (A, D) if D - F <= A else None,
        (B, C) if C - E >= B else None,
        (C - E, C) if B > C - E > A else None,
        (C - F, C) if C - F > A else None,
        (A, C) if C - F <= A <= C - E else None
    ]
    area_points = (points + [
        (A, 0) if _between(0, lines[0]) else None,
        (B, 0) if _between(0, lines[1]) else None,
        (0, D) if _between(0, lines[2]) else None,
        (0, C) if _between(0, lines[3]) else None,
        (-F / 2, F / 2) if _between(-F / 2, lines[4]) else None,
        (-E / 2, E / 2) if _between(-E / 2, lines[5]) else None
    ])
    # print(tuple(filter(None, area_points)))
    e1_, e2_ = min(filter(None, area_points), key=lambda x: abs(x[0])+abs(x[1]))  # these make min(|e1|+|e2|)
    angle_points = (points + [
        (A, A) if _between(A, lines[0]) else None,
        (A, -A) if _between(-A, lines[0]) else None,
        (B, B) if _between(B, lines[1]) else None,
        (B, -B) if _between(-B, lines[1]) else None,
        (D, D) if _between(D, lines[2]) else None,
        (D, -D) if _between(-D, lines[2]) else None,
        (C, C) if _between(C, lines[3]) else None,
        (C, -C) if _between(-C, lines[3]) else None,
        (-F / 2, F / 2) if _between(-F / 2, lines[4]) else None,
        (-E / 2, E / 2) if _between(-E / 2, lines[5]) else None
    ])
    # print(tuple(filter(None, angle_points)))
    e1__, e2__ = min(sorted(filter(None, angle_points), key=lambda x: abs(x[0])+abs(x[1])),
                     key=lambda x: min(abs(x[0] + x[1]), abs(x[0] - x[1])))  # these make min(||e1|-|e2||)
    e1, e2 = (e1_ + e1__) / 2, (e2_ + e2__) / 2
    print(e1_, e2_)
    print(e1__, e2__)
    return e1 + G, e2 + H
    # for ae2 in np.arange(0, max(map(abs, e2_area)), 0.25):                  # these make min(|e2|)
    #     for ae1 in np.arange(0, max(map(abs, e1_area)), 0.25):
    #         for e2 in -ae2, ae2:
    #             if C<=e2<=D:
    #                 for e1 in ae1, -ae1:
    #                     if A<=e1<=B:
    #                         if E<=e2 - e1<=F:
    #                             return e1, e2
    # return A, D


def _phantom_base(c1, c2, bk0, bk1):
    D = np.subtract(c2, c1, dtype=np.int16)
    d_min = D.min()     # (e1 - e2) / k
    d_max = D.max()     # (b - a) / k + c_min
    c1_min, c1_max = c1.min(), c1.max()
    c2_max, c2_min = c2.max(), c2.min()
    print(d_min, d_max, c1_min, c2_max)
    k = min(255 / (c2_max - c1_min - d_min), 255 / (c1_max - c1_min), 255 / (c2_max - c2_min))
    de_area = (- k * d_min, 255 - k * d_max)#sorted
    e1_area = (- k * c1_min, 255 - k * c1_max)
    e2_area = (- k * c2_min, 255 - k * c2_max)
    set_point = (1 - k) * 255/2
    # print('k', 255 / (c2_max - c1_min - d_min), 255 / (c1_max - c1_min), 255 / (c2_max - c2_min))
    print('e1', (- k * c1_min, 255 - k * c1_max))
    print('e2', (- k * c2_min, 255 - k * c2_max))
    print('de', (- k * d_min, 255 - k * d_max))
    e1, e2 = optimiz(*e1_area, *e2_area, *de_area, set_point, set_point)
    de = e2 - e1
    print(k, e1, e2)
    x1 = c1 * k + e1
    # x2 = c2 * k + e2
    # print(x1, x2)
    a = (k * D + de) / (bk1 - bk0)
    q = 1 - a
    p = np.around((x1-bk0*a)/np.clip(q,0.0001,1)).astype(np.uint8)
    q *= 255; np.around(q, out=q); q=q.astype(np.uint8)
    # print('q', q)
    # print('p', p)
    return p, q

_div = lambda a, b: a / b if b else math.copysign(math.inf, a)

def phantom_tank_chess(img0: IMGStruct, img1: IMGStruct, bk0=0, bk1=255):
    """:TODO: Merge two clourful picture in a cross chess board, it will fill with neutral gray.
    :NOTO: It can't reveal color precisely, and cound leak the background color. If merge two gray figure, it's not nesscessary."""
    assert bk0 < bk1, "If what to get such a different background, Please swap image 0 and image 1"
    assert img0.img.size == img1.img.size
    c1 = np.asarray(img0.img, dtype=np.float32)
    c2 = np.asarray(img1.img, dtype=np.float32)
    md = (bk0 + bk1) / 2
    H1 = (bk1 - md) / bk1 #max(, _div((bk1 - md), (bk1 - 255)), 0)
    H2 = (md - bk0) / (255 - bk0) #max(_div((bk0 - md), bk0), , 0)
    L1 = np.clip((md + (bk0 - bk1) * (1 - H1), md), 0, 255)
    L2 = np.clip((md, md - (bk0 - bk1) * (1 - H2)), 0, 255)
    # print(L1, L2)

    # print(c1[0, :10].astype(np.uint8))
    # print(c2[0, :10].astype(np.uint8))
    g1, g2 = c1.mean(axis=-1, keepdims=True), c2.mean(axis=-1, keepdims=True)
    for i,(a,b) in enumerate(zip(g1, g2)):
        a[i%2::2] = md
        b[(i+1)%2::2] = md
    l1, h1 = g1.min(), g1.max()
    l2, h2 = g2.min(), g2.max()
    c1 *= (L1[1] - L1[0]) / (h1 - l1); c1 += l1
    c2 *= (L2[1] - L2[0]) / (h2 - l2); c2 += l2
    g1 *= (L1[1] - L1[0]) / (h1 - l1); g1 += l1
    g2 *= (L2[1] - L2[0]) / (h2 - l2); g2 += l2
    # print(l1, h1, l2, h2)
    for i,(a,b) in enumerate(zip(c1, c2)):
        a[i%2::2] = md
        b[(i+1)%2::2] = md

    q1 = 1 - (g1 - md) / (bk0 - bk1)    #q1>H1
    p1 = (c1 - bk0) / q1 + bk0 #(md - bk1) / q1 + bk1 #
    q2 = 1 - (md - g2) / (bk0 - bk1)    #q2>H2
    p2 = (c2 - bk1) / q2 + bk1 #(md - bk0) / q2 + bk0
    np.around(p1, out=p1)
    np.around(p2, out=p2)
    q1 *= 255; np.around(q1, out=q1)
    q2 *= 255; np.around(q2, out=q2)
    p1 = np.clip(p1, 0, 255, out=p1).astype(np.uint8)
    p2 = np.clip(p2, 0, 255, out=p2).astype(np.uint8)
    q1 = np.clip(q1, 0, 255, out=q1).astype(np.uint8)
    q2 = np.clip(q2, 0, 255, out=q2).astype(np.uint8)
    x1 = np.concatenate([p1, q1], axis=-1)
    x2 = np.concatenate([p2, q2], axis=-1)
    for i,a in enumerate(x1):
        a[i%2::2] = x2[i, i%2::2]

    # print('q1',q1[0, :10].astype(np.uint8))
    # print('q2',q2[0, :10].astype(np.uint8))
    # print('p1',p1[0, :10].astype(np.uint8))
    # print('p2',p2[0, :10].astype(np.uint8))
    # print('x1',x1[0, :10].astype(np.uint8))
    # print('x2',x2[0, :10].astype(np.uint8))
    img = Image.fromarray(x1, mode='RGBA')
    return img

def paste_image(img, background, center=(0, 0)):
    assert (img.img.size[0] <= background.img.size[0] and img.img.size[1] <= background.img.size[1])
    k = min(background.img.size[0] / img.img.size[0], background.img.size[1] / img.img.size[1])
    if k!=1:
        img.img = img.img.resize((int(k * img.img.size[0]), int(k * img.img.size[1])))

    center = (background.img.size[0] - img.img.size[0] >> 1) + center[0], (background.img.size[1] - img.img.size[1] >> 1) + center[1]
    # print(img.img.size, background.img.size, center)

    background.img.paste(img.img, box=center, mask=img.img.split()[3] if img.img.mode == 'RGBA' else None)
    background.arr = np.asarray(background.img)


def phantom_tank_weigth(new: IMGStruct, img0: IMGStruct, img1: IMGStruct, bk0=0, bk1=255, alpha=0):
    """:TODO: Merge two colorful picture by using a weight. \alpha nore near to zero then more color behind background0 was remained, otherwise background1.
    \alpha = \sum{| p*q + b2*(1-q) - rgb_1 |} / \sum{| rgb_2 - rgb_1 |}"""
    assert bk0 < bk1, "If what to get such a different background, Please swap image 0 and image 1"
    assert img0.img.size == img1.img.size
    assert 0<= alpha <=1
    c1 = np.asarray(img0.img, dtype=np.float32)
    c2 = np.asarray(img1.img, dtype=np.float32)
    D = c2.mean(axis=-1, keepdims=True) - c1.mean(axis=-1, keepdims=True)
    c1 = (1 - alpha) * c1 + alpha * (c2 - D)
    c2 = c1 + D
    d_min, d_max = D.min(), D.max()
    c1_min, c1_max = c1.min(), c1.max()
    c2_max, c2_min = c2.max(), c2.min()
    print(d_min, d_max, c1_min, c2_max)
    k = min(255 / (c2_max - c1_min - d_min), 255 / (c1_max - c1_min), 255 / (c2_max - c2_min))
    de_area = (- k * d_min, 255 - k * d_max)#sorted
    e1_area = (- k * c1_min, 255 - k * c1_max)
    e2_area = (- k * c2_min, 255 - k * c2_max)
    set_point = (1 - k) * 255/2
    # print('k', 255 / (c2_max - c1_min - d_min), 255 / (c1_max - c1_min), 255 / (c2_max - c2_min))
    print('e1', (- k * c1_min, 255 - k * c1_max))
    print('e2', (- k * c2_min, 255 - k * c2_max))
    print('de', (- k * d_min, 255 - k * d_max))
    e1, e2 = optimiz(*e1_area, *e2_area, *de_area, set_point, set_point)
    de = e2 - e1
    print(k, e1, e2)
    x1 = c1 * k + e1
    # x2 = c2 * k + e2
    # print(x1, x2)
    a = (k * D + de) / (bk1 - bk0)
    q = 1 - a
    p = np.around((x1-bk0+bk0*q)/np.clip(q,0.0001,1))#(bk1 * x1 - bk0 * x2) / w)
    p = np.clip(p, 0, 255, out=p).astype(np.uint8)
    q *= 255; np.around(q, out=q); np.clip(q, 0, 255, out=q)
    q = q.astype(np.uint8)
    new.close()
    new.img = Image.fromarray(np.concatenate([p, q], axis=-1), mode='RGBA')
    return new.img


if __name__ == "__main__":
    # path0 = r'D:\picture\EditedPhotos\Screenshot_2022-01-27-11-34-22.jpg'
    # path1 = r'D:\picture\EditedPhotos\Screenshot_2022-01-27-13-31-44.jpg'
    path0 = r'D:\picture\dd.png'
    path1 = r'D:\picture\aq.jpg'
    assert os.path.isfile(path0) and os.path.isfile(path1)
    dir0 = os.path.dirname(path0)
    dir1 = os.path.dirname(path1)
    # with IMGStruct(Image.open(path0)) as img0:
    #     arr0s = img0.img.split()
    #     name0 = os.path.splitext(path0)
    #     for i in range(len(arr0s)):
    #         with IMGStruct(arr0s[i]) as img0i:
    #             img0i.save(os.path.join(dir0, f'{name0[0]}-alpha-{i}{name0[1]}'))
    #
    # with IMGStruct(Image.open(path1)) as img1:
    #     arr1s = img1.img.split()
    #     name1 = os.path.splitext(path1)
    #     for i in range(len(arr1s)):
    #         with IMGStruct(arr1s[i]) as img1i:
    #             img1i.save(os.path.join(dir1, f'{name1[0]}-alpha-{i}{name1[1]}'))
    #
    # path0 = path0.rsplit('.', maxsplit=1)[0] +'-alpha-1.jpg'
    # path1 = path1.rsplit('.', maxsplit=1)[0] +'-alpha-1.jpg'
    assert os.path.isfile(path0) and os.path.isfile(path1)
    dir0 = os.path.commonpath((path0, path1))
    a = 0
    b = 255
    alpha = 0.5
    with IMGStruct(Image.open(path0)) as img0:
        with IMGStruct(Image.open(path1)) as img1:

            size = max(img0.img.size[0], img1.img.size[0]), max(img1.img.size[1], img0.img.size[1])
            with IMGStruct(Image.new('RGB', size, ((a+b)//2,)*3), 0) as img0n:
                with IMGStruct(Image.new('RGB', size, ((a+b)//2,)*3), 0) as img1n:
                    paste_image(img0, img0n)
                    paste_image(img1, img1n)
                    # new = phantom_tank_chess(img0n, img1n, a, b)
                    with IMGStruct(Image.new('RGBA', size)) as new:
                        new = phantom_tank_weigth(new, img0n, img1n, a, b, alpha)
                        new.save(os.path.join(dir0,
    f'{os.path.splitext(os.path.basename(path0))[0]}-{os.path.splitext(os.path.basename(path1))[0]}.png'))
    #         with IMGStruct(Image.new('L', size), 0) as img0n:
    #             with IMGStruct(Image.new('L', size), 0) as img1n:
    #                 paste_image(img0, img0n)
    #                 paste_image(img1, img1n)
    #
    #                 with IMGStruct(Image.new('RGBA', size)) as new:
    #                     phantom_tank(new, img0n, img1n, a, b)
    #                     new.img.save(os.path.join(dir0,
    # f'{os.path.splitext(os.path.basename(path0))[0]}-{os.path.splitext(os.path.basename(path1))[0]}.png')
    #                                 )

def optimiz(A,B,C,D,E,F,G=0,H=0):
    A -= G; B -= G; C -= H; D -= H
    if A<=0<=B:
        if C<=0<=D:
            if E<=0<=F:
                return G, H
    points = [
        (A, D) if F+A >= D else None,
        (A, F+A) if D > F+A > C else None,
        (A, E+A) if E+A > C else None,
        (A, C) if E+A <= C <= F+A else None,
        (B, D) if F+B >= D >= E+B else None,
        (B, F+B) if D > F+B else None,
        (B, E+B) if D > E+B > C else None,
        (B, C) if E+B <= C else None,
        (B, D) if D-E >= B >= D-F else None,
        (D-E, D) if B > D-E else None,
        (D-F, D) if B > D-F > A else None,
        (A, D) if D-F <= A else None,
        (B, C) if C-E >= B else None,
        (C-E, C) if B > C-E > A else None,
        (C-F, C) if C-F > A else None,
        (A, C) if C-F <= A <= C-E else None
    ]
    lines = [
        (max(C,A+E),min(D,A+F)),        #e1 = A
        (max(C,B+E),min(D,B+F)),        #e1 = B
        (max(A,D-F),min(B,D-E)),        #e2 = D
        (max(A,C-F),min(B,C-E)),        #e2 = C
        (max(A,C-F),min(B,D-F)),        #e2 = e1 + F
        (max(C-E,A),min(B,D-E))         #e2 = e1 + E
    ]
    area_points = (points + [
        (A, 0) if _between(0, lines[0]) else None,
        (B, 0) if _between(0, lines[1]) else None,
        (0, D) if _between(0, lines[2]) else None,
        (0, C) if _between(0, lines[3]) else None,
        (-F/2, F/2) if _between(-F/2, lines[4]) else None,
        (-E/2, E/2) if _between(-E/2, lines[5]) else None
    ])
    print(tuple(filter(None, area_points)))
    e1_, e2_ = min(filter(None, area_points), key=lambda x: x[0]**2+x[1]**2)       # these min(|e1|+|e2|)
    if _between(A, lines[0]): points.append((A, A))
    if _between(-A, lines[0]): points.append((A, -A))
    if _between(B, lines[1]): points.append((B, B))
    if _between(-B, lines[1]): points.append((B, -B))
    if _between(D, lines[2]): points.append((D, D))
    if _between(-D, lines[2]): points.append((D, -D))
    if _between(C, lines[3]): points.append((C, C))
    if _between(-C, lines[3]): points.append((C, -C))
    if _between(-F/2, lines[4]): points.append((-F/2, F/2))
    if _between(-E/2, lines[5]): points.append((-E/2, E/2))
    print(tuple(filter(None, points)))
    e1__, e2__ = min(sorted(filter(None, points), key=lambda x: x[0]**2+x[1]**2),
                     key=lambda x: min(abs(x[0]+x[1]), abs(x[0]-x[1])))       # these min(||e1|-|e2||)
    e1, e2 = (e1_ + e1__) / 2, (e2_ + e2__) / 2
    print(e1_, e2_)
    print(e1__, e2__)
    return e1 + G, e2 + H
