import numpy as np
import os, sys
import imageio


def rgb2gray(rgb, normal=np.array([[[0.2989, 0.5870, 0.1140]]], dtype=np.float32)):
    rgb = rgb.astype(normal.dtype)
    rgb *= normal
    return rgb.sum(axis=-1)


def phantom_tank(new, img0, img1, bk0=0, bk1=255):
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
    assert img0.shape == img1.shape == new.shape
    p, q = _phantom_base(img0, img1, bk0, bk1)
    new[..., 0] = new[:, :, 1] = new[:, :, 2] = p
    new[:, :, 3] = q
    return new


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
    a = (k * D + de) / (bk1 - bk0)
    q = 1 - a
    p = np.around((x1-bk0*a)/np.clip(q,0.0001,1)).astype(np.uint8)
    q *= 255; np.around(q, out=q); q=q.astype(np.uint8)
    return p, q


def paste_image(img, background, center=(0, 0)):
    b_c = background.shape[1]//2, background.shape[0]//2
    i_c = img.shape[1]//2, img.shape[0]//2
    pos = b_c[0]+center[0]-i_c[0], b_c[1]+center[1]-i_c[1]
    pad = (max(0, - pos[1]), max(0, pos[1]+img.shape[0]-background.shape[0])),\
          (max(0, - pos[0]), max(0, pos[0]+img.shape[1]-background.shape[1]))
    print(b_c, i_c, pos, pad)
    background[max(0, pos[1]): max(0, pos[1]+img.shape[0]), max(0, pos[0]): max(0, pos[0]+img.shape[1])] = \
           img[pad[0][0]: max(0, img.shape[0]-pad[0][1]), pad[1][0]: max(0, img.shape[1]-pad[1][1])]

def phantom_image(path0, path1, dir0=None, a=0, b=255, center0=None, center1=None, scale=1):
    assert os.path.isfile(path0), f'path0 {path0} is not a file.'
    assert os.path.isfile(path1), f'path1 {path1} is not a file.'
    assert 0 <= a < b <= 255
    if not dir0:
        dir0 = os.path.commonpath((path0, path1))
    assert os.path.exists(dir0), f'dir0 {dir0} is not dirt.'
    img0 = imageio.imread_v2(path0)
    img1 = imageio.imread_v2(path1)

    if img0.ndim == 3:
        if img0.shape[2] == 4:
            img0 = img0[:, :, :3]
        img0 = rgbToGray(img0)
    if img1.ndim == 3:
        if img1.shape[2] == 4:
            img1 = img1[:, :, :3]
        img1 = rgbToGray(img1)
    if scale != 1:
        scale, img0, img1 = scale_image(img0, img1, scale)
    size = max(img0.shape[1], img1.shape[1]), max(img1.shape[0], img0.shape[0])
    img0n = np.full((size[1], size[0]), (a+b)//2, dtype=np.uint8)
    img1n = np.full((size[1], size[0]), (a+b)//2, dtype=np.uint8)
    if center0:
        paste_image(img0, img0n, center0)
    else:
        paste_image(img0, img0n)
    if center1:
        paste_image(img1, img1n, center1)
    else:
        paste_image(img1, img1n)

    new = np.empty((size[1], size[0], 4), dtype=np.uint8)
    phantom_tank(new, img0n, img1n, a, b)
    imageio.imwrite(os.path.join(dir0,
    f'{os.path.splitext(os.path.basename(path0))[0][:20]}-{os.path.splitext(os.path.basename(path1))[0][:20]}.png'),
        new)
    return scale

def scale_image(img0, img1, scale=0):
    assert 0 <= scale
    if scale == 1:
        return img0, img1
    size0 = img0.size
    size1 = img1.size
    shape0 = img0.shape
    shape1 = img1.shape
    bigger = (size0 > size1)
    if scale == 0:
        scale = np.sqrt(size1 / size0 if bigger else size0 / size1)
    else:
        bigger = (scale < 1)
        if not bigger:
            scale = 1 / scale

    size0 = (int(shape0[1]*scale) if bigger else shape0[1], int(shape0[0]*scale) if bigger else shape0[0])
    size1 = (shape1[1] if bigger else int(shape1[1]*scale), shape1[0] if bigger else int(shape1[0]*scale))
    size = (max(size0[0], size1[0]), max(size0[1], size1[1]))
    from PIL import Image
    img = img0 if bigger else img1
    shape = shape0 if bigger else shape1
    image = Image.new(('RGB' if img.ndim==3 else 'L'), (shape[1], shape[0]))
    img.resize(img.size)
    image.putdata(img)
    img.resize(shape)
    new = image.resize(size0 if bigger else size1)
    img = np.asarray(new)
    image.close()
    new.close()
    return (scale if bigger else 1/scale), (img if bigger else img0), (img1 if bigger else img)

usage = '''\
    usage:
        python phantom_tank.py [-a int<0-254>] [-b int<1-255>] [-q] [-d <file folder>] <img0|img1> ...
        -a: (dauflt 0) set the background color while img0 shuld be show.
        -b: (dauflt 255) set the background color while img1 shuld be show.
        -d: (dauflt .) set the path to save the outcomes.
        -q: (dauflt false) dicede wether pass through errors.
        -s: (dauflt 0) control the relative size of images, 0 means auto resize, 1 means do nothing, 
        lesser 1 means 
        Images should divide as couples, while the front one to show as img0 which is you want,
        and they were separate by | character without any space.'''

def cmd_strip(message, start):
    if message.startswith(start):
        message = message[len(start):]
        if message[0].isspace():
            return message.lstrip()

if __name__ == '__main__':
    a = 0
    b = 255
    d = '.'
    scale = 0
    print(sys.argv)
    if len(sys.argv) <= 1:
        print(usage)
        path0 = path1 = ''
        center0 = center1 = (0,0)
        while True:
            message = input('> ').strip()
            if message == 'exit':
                break
            r = cmd_strip(message, '-a')
            if r:
                a = int(r); continue
            r = cmd_strip(message, '-b')
            if r:
                b = int(r); continue
            r = cmd_strip(message, '-d')
            if r:
                d = r
                continue
            r = cmd_strip(message, '--path0')
            if r:
                path0 = r.replace('<>', d)
                continue
            r = cmd_strip(message, '--path1')
            if r:
                path1 = r.replace('<>', d)
                continue
            r = cmd_strip(message, '--center0')
            if r:
                center0 = eval(r)
                continue
            r = cmd_strip(message, '--center1')
            if r:
                center1 = eval(r)
                continue
            r = cmd_strip(message, '-s')
            if r:
                scale = float(r)
                continue
            if message =='create':
                try:
                    phantom_image(path0, path1, d, a, b, center0, center1, scale)
                except AssertionError as message:
                    print(message)
                print(f'''\
    {path0=}
    {path1=}
    folder={d}
    {a=} {b=} {scale=:.3f} {center0=} {center1=}''')
    else:
        quit = False
        argv = sys.argv[1:]
        if '--' in argv:
            argv.remove('--')
        if '-a' in argv:
            i = argv.index('-a')
            del argv[i]
            a = eval(argv.pop(i))
        if '-b' in argv:
            i = argv.index('-b')
            del argv[i]
            b = eval(argv.pop(i))
        if '-d' in argv:
            i = argv.index('-d')
            del argv[i]
            d = argv.pop(i)
        if '-q' in argv:
            argv.remove('-q')
            quit = True
        if '-s' in argv:
            i = argv.index('-s')
            del argv[i]
            scale = eval(argv.pop(i))

        if not quit:
            print(usage)
        li = []
        for p in argv:
            li.append(p.split('='))

        for path0, path1 in li:
            try:
                phantom_image(path0.replace('<>', d), path1.replace('<>', d), d, a, b, scale=scale)
            except:
                if not quit:
                    raise
                print(f'{path0} | {path1}')
