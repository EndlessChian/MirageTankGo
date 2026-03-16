let File0 = document.querySelector('#form0>input');
let File1 = document.querySelector('#form1>input');
let img0 = document.querySelector('#image0>img');
let img1 = document.querySelector('#image1>img');
let out = document.querySelector('#output>img');
let cls0 = document.querySelector('#form0');
let cls1 = document.querySelector('#form1');
let color0 = document.querySelector('#background0');
let color1 = document.querySelector('#background1');
let color_convert = document.querySelector('#convert');
let can0 = document.querySelector('#image0>canvas');
let can1 = document.querySelector('#image1>canvas');
let can2 = document.querySelector('#output>canvas');

// 获取新增控件
const weightMode = document.getElementById('weightMode');
const weightSlider = document.getElementById('weightSlider');
const weightValue = document.getElementById('weightValue');

let ctx0, ctx1, ctx2;
let color_mode = false;

const lookImage = (input) => {

    if(input.files.length){
        var file = input.files[0];
        if(file.type.search('image/') == 0){
            var filereader = new FileReader();
            filereader.onload = function(evt){
                if (filereader.readyState == 2){
                    var i = document.querySelector('#' + input.title + '>img');
                    i.src = evt.target.result;
                    i.onload = function(){
                        var scale = document.querySelector('#' + input.title + '>form>div.alpha').offsetWidth / i.naturalWidth;
                        var ctx = document.querySelector('#' + input.title + '>canvas');
                        ctx.width = i.naturalWidth;
                        ctx.height = i.naturalHeight;
                        ctx.getContext('2d').drawImage(i, 0, 0);

                        i.style.width = parseInt(scale * i.naturalWidth) + 'px';
                        i.style.height = parseInt(scale * i.naturalHeight) + 'px';
                        var mask = document.querySelector('#' + input.title + '>span.mask');
                        mask.style.backgroundImage = `url(${i.src})`;
                        i.onload = null;
                    }
                }
            }
            filereader.readAsDataURL(file);
        }
    }
}

function gray(input){
    if (input.files.length == 0)
        return;
    var im = input.title.at(-1), li = [];
    if ((im == 0 && !( img0.height && img0.width))||
        ((im == 1 && !( img1.height && img1.width)))) return console.log('Error image'+im+' size.');

    var cls = (im == 0? cls0: cls1).classList;
    cls.forEach(c =>
        li.push(c == 'red'? 0: c == 'green'? 1: 2)
    );
    if (li.length == 0){
        console.log('No image'+im+' colors.');
        return;
//        cls.value = 'red green blue';
//        li = [0, 1, 2];
    }
    var ctx = document.querySelector('#image'+im+'>canvas');
    var data = ctx.getContext('2d').getImageData(0, 0, ctx.width, ctx.height).data;
    var idata = new ImageData(ctx.width, ctx.height);
    var arr = idata.data;
    arr.fill(255);
    for (var i=0, j=data.length, k=0, u=li.length; i < j; i++,k=0){
        li.forEach(c => k += data[i + c]);
        arr[i++] = arr[i++] = arr[i++] = parseInt(Math.round(k/u));
    }
    var ictx = document.querySelector("#output>canvas");
    ictx.width = ctx.width;
    ictx.height = ctx.height;
    ictx.getContext('2d').putImageData(idata, 0, 0);
    (im == 0? img0: img1).src = ictx.toDataURL('image/png');
}

const between = (x, ab) => ab[0]<x && x<ab[1];

function optimiz(A,B,C,D,E,F,G,H){
    A -= G; B -= G; C -= H; D -= H;
    if (A <= 0 && 0 <= B){
        if (C <= 0 && 0 <= D)
            if (E <= 0 && 0 <= F)
                return [G, H];
    }
    var lines = [
        [Math.max(C, A + E), Math.min(D, A + F)],  //# e1 = A
        [Math.max(C, B + E), Math.min(D, B + F)],  //# e1 = B
        [Math.max(A, D - F), Math.min(B, D - E)],  //# e2 = D
        [Math.max(A, C - F), Math.min(B, C - E)],  //# e2 = C
        [Math.max(A, C - F), Math.min(B, D - F)],  //# e2 = e1 + F
        [Math.max(C - E, A), Math.min(B, D - E)]  //# e2 = e1 + E
    ];
    var points = [
        (F + A >= D)? [A, D]: null,
        between(F+A, [C, D])? [A, F + A]: null,
        (E + A > C)? [A, E + A]: null,
        (E <= C-A && C-A <= F)? [A, C]: null,
        (E <= D-B && D-B <= F)? [B, D]: null,
        (D > F + B)? [B, F + B]: null,
        between(E + B, [C, D])? [B, E + B]: null,
        (E + B <= C)? [B, C]: null,
        (D - E >= B && B >= D - F)? [B, D]: null,
        (B > D - E)? [D - E, D]: null,
        between(D - F, [A, B])? [D - F, D]: null,
        (D - F <= A)? [A, D]: null,
        (C - E >= B)? [B, C]: null,
        between(C - E, [A, B])? [C - E, C]: null,
        (C - F > A)? [C - F, C]: null,
        (C - F <= A && A <= C - E)? [A, C]: null
    ];
    var area_points = points.concat([
        between(0, lines[0])? [A, 0]: null,
        between(0, lines[1])? [B, 0]: null,
        between(0, lines[2])? [0, D]: null,
        between(0, lines[3])? [0, C]: null,
        between(-F / 2, lines[4])? [-F / 2, F / 2]: null,
        between(-E / 2, lines[5])? [-E / 2, E / 2]: null
    ]).filter(n => n !== null);
    console.log(area_points);
    var ee_=area_points[0], e_min=1000;
    area_points.forEach((x) => {
        var ne = Math.abs(x[0])+Math.abs(x[1]);
        if (ne < e_min){
            ee_ = x;
            e_min = ne;
        }
    });
    var e1_ = ee_[0], e2_ = ee_[1];
    var angle_points = points.concat([
        between(A, lines[0])? [A, A]: null,
        between(-A, lines[0])? [A, -A]: null,
        between(B, lines[1])? [B, B]: null,
        between(-B, lines[1])? [B, -B]: null,
        between(D, lines[2])? [D, D]: null,
        between(-D, lines[2])? [D, -D]: null,
        between(C, lines[3])? [C, C]: null,
        between(-C, lines[3])? [C, -C]: null,
        between(-F / 2, lines[4])? [-F / 2, F / 2]: null,
        between(-E / 2, lines[5])? [-E / 2, E / 2]: null
    ]).filter(n => n !== null);
    console.log(angle_points);
    var e__min=1000;
    ee_=angle_points[0], e_min=1000;
    angle_points.forEach((x) => {
        var ne = Math.min(Math.abs(x[0] + x[1]), Math.abs(x[0] - x[1]));
        var me = Math.abs(x[0])+Math.abs(x[1]);
        if (ne < e_min-0.1 || (ne < e_min+0.1 && me < e__min)){
            ee_ = x;
            e_min = ne;
            e__min = me;
        }
    });
    var e1__ = ee_[0], e2__ = ee_[1];
    var e1 = (e1_ + e1__) / 2, e2 = (e2_ + e2__) / 2;
    console.log(e1_, e2_);
    console.log(e1__, e2__);
    return [e1 + G, e2 + H];
}

function minmax(arr){
    var min = arr[0], max = arr[0];
    arr.forEach((n) => {
        if (n > max) max = n;
        else if (n < min) min = n;
    })
    return [min, max];
}

function phantom_tank(bk, c1, c2, bk0, bk1, D){
    var c1_min, c1_max, c2_min, c2_max;
    var d_min, d_max;
    c1_min = c1_max = c1[0];
    c2_min = c2_max = c2[0];
    d_min = d_max = D[0];
    D.forEach((n, i) => {
        if (n > d_max) d_max = n;
        else if (n < d_min) d_min = n;
        n = c1[i<<2];
        if (n > c1_max) c1_max = n;
        else if (n < c1_min) c1_min = n;
        n = c2[i<<2];
        if (n > c2_max) c2_max = n;
        else if (n < c2_min) c2_min = n;
    })
    console.log(d_min, d_max, c1_min, c2_max);

    const k = Math.min(255 / (c2_max - c1_min - d_min), 255 / (c1_max - c1_min), 255 / (c2_max - c2_min));
    const de_area = [- k * d_min, 255 - k * d_max];
    const e1_area = [- k * c1_min, 255 - k * c1_max];
    const e2_area = [- k * c2_min, 255 - k * c2_max];
    const set_point = (1 - k) * 255/2;
    console.log('e1', e1_area);
    console.log('e2', e2_area);
    console.log('de', de_area);
    var e1, e2, ee, de;
    ee = optimiz(...e1_area, ...e2_area, ...de_area, set_point, set_point);
    e1 = ee[0]; e2 = ee[1];
    de = e2 - e1;
    console.log(k, e1, e2);
//    const x1 = c1.map(n => n*k + e1);
//    var x2 = c2.map(n => n*k + e2);

    const db = 1 / (bk1 - bk0);
    var q, p, a;
    D.forEach((d, i) => {
        a = (k * d + de) * db;
        q = 1 - a;
        i<<=2;
        p = Math.round((k * c1[i] + e1 - bk0 * a) / Math.max(db, q));
        bk[i++] = bk[i++] = bk[i++] = p;
        bk[i] = Math.round(q * 255);
    });
}

function phantom_tank_color(bk, c1, c2, bk0, bk1, D){
    var c1_min, c1_max, c2_min, c2_max;
    c1_min = c1_max = c1[0];
    c1.forEach((n, i) => {
        if ((i+1) & 3)
            if (n > c1_max) c1_max = n;
            else if (n < c1_min) c1_min = n;
    })
    c2_min = c2_max = c2[0];
    c2.forEach((n, i) => {
        if ((i+1) & 3)
            if (n > c2_max) c2_max = n;
            else if (n < c2_min) c2_min = n;
    })
    var d_min, d_max;
    d_min = d_max = D[0];
    D.forEach((n) => {
        if (n > d_max) d_max = n;
        else if (n < d_min) d_min = n;
    })
    console.log(d_min, d_max, c1_min, c2_max);

    const k = Math.min(255 / (c2_max - c1_min - d_min), 255 / (c1_max - c1_min), 255 / (c2_max - c2_min));
    const de_area = [- k * d_min, 255 - k * d_max];
    const e1_area = [- k * c1_min, 255 - k * c1_max];
    const e2_area = [- k * c2_min, 255 - k * c2_max];
    const set_point = (1 - k) * 255/2;
    console.log('e1', e1_area);
    console.log('e2', e2_area);
    console.log('de', de_area);
    var e1, e2, ee, de;
    ee = optimiz(...e1_area, ...e2_area, ...de_area, set_point, set_point);
    e1 = ee[0]; e2 = ee[1];
    de = e2 - e1;
    console.log('k', k, 'e1', e1, 'e2', e2);
//    const x1 = c1.map(n => n*k + e1);
//    var x2 = c2.map(n => n*k + e2);

    const db = 1 / (bk1 - bk0);
    var q, a, _q;
    D.forEach((d, i) => {
        a = (k * d + de) * db;
        q = 1 - a;
        a = e1 - bk0 * a;
        i <<= 2;
        // p = (b2*a1 - b1*a2) / ((b2 - a2) + (a1 - b1))
        if (q >= db){
            _q = 1 / q;
        } else {
            _q = bk1 - bk0;
        }
        bk[i] = Math.round((k * c1[i] + a) * _q); i++;
        bk[i] = Math.round((k * c1[i] + a) * _q); i++;
        bk[i] = Math.round((k * c1[i] + a) * _q); i++;
        bk[i] = Math.round(q * 255);
    });
}

function tank_create(){
    if(File0.files.length == 0 || File1.files.length == 0 || ! img0.src || ! img1.src ||
        ! img0.naturalWidth || ! img1.naturalWidth || ! img0.naturalHeight || ! img1.naturalHeight)
        return console.log('onclick: worry Image type.');
    var a = parseInt(color0.value), b = parseInt(color1.value), mid = (a+b)>>1;
    if(isNaN(a) || isNaN(b) || a < 0 || a >= b || b > 255)
        return console.log('Color dis-excepted.', a, b);
    var alpha = parseFloat(weightSlider.value), beta=1-alpha;
    if(isNaN(alpha) || alpha < 0 || alpha > 1)
        return console.log('Weight dis-excepted.', alpha);
    var size0 = [img0.naturalWidth, img0.naturalHeight], s0 = size0[0]*size0[1],
     size1 = [img1.naturalWidth, img1.naturalHeight], s1 = size1[0]*size1[1],
      scale0 = Math.sqrt((s1/s0 + 1)/2), scale1 = Math.sqrt((s0/s1 + 1)/2),
    size = [
        parseInt((size0[0]+size1[0])/2),
        parseInt((size0[1]+size1[1])/2)
    ];
    size0 = size0.map(c=>parseInt(c*scale0));
    size1 = size1.map(c=>parseInt(c*scale1));
    var ctx = document.querySelector("#output>canvas");
    ctx.width = size[0];
    ctx.height = size[1];
    var c = ctx.getContext('2d', { willReadFrequently: true });
    // 填充背景
    c.fillStyle = `rgb(${mid},${mid},${mid})`;
    c.fillRect(0, 0, ...size);
    c.drawImage(img0, (size[0]-size0[0])>>1, (size[1]-size0[1])>>1, ...size0);
    var arr0 = c.getImageData(0, 0, ...size).data, data0 = arr0;
    if (color_mode){
        c.drawImage(can0,
            0, 0, img0.naturalWidth, img0.naturalHeight,
            (size[0]-size0[0])>>1, (size[1]-size0[1])>>1, ...size0);
        data0 = c.getImageData(0, 0, ...size).data;
    }

    c.fillRect(0, 0, ...size);
    c.drawImage(img1, (size[0]-size1[0])>>1, (size[1]-size1[1])>>1, ...size1);
    var arr1 = c.getImageData(0, 0, ...size).data, data1 = arr1;
    if (color_mode){
        c.drawImage(can1,
            0, 0, img1.naturalWidth, img1.naturalHeight,
            (size[0]-size1[0])>>1, (size[1]-size1[1])>>1, ...size1);
        data1 = c.getImageData(0, 0, ...size).data;
    }

    var D = new Int16Array(size[0]*size[1]);
    for (var i=0, j=D.length; i<j; i++) D[i] = arr1[i<<2] - arr0[i<<2];
    if (color_mode){
        D.forEach((d, i) => {
            var p;
            i <<= 2;
            p = beta * data0[i] + alpha * (data1[i] - d);
            data0[i] = Math.round(p);
            data1[i++] = Math.round(p + d);
            p = beta * data0[i] + alpha * (data1[i] - d);
            data0[i] = Math.round(p);
            data1[i++] = Math.round(p + d);
            p = beta * data0[i] + alpha * (data1[i] - d);
            data0[i] = Math.round(p);
            data1[i] = Math.round(p + d);
        })
    }

    var data = c.getImageData(0, 0, ...size);
    if (color_mode){
        phantom_tank_color(data.data, data0, data1, a, b, D);      // Uint8ClampedArray
    } else {
        phantom_tank(data.data, arr0, arr1, a, b, D);      // Uint8ClampedArray
    }
    c.putImageData(data, 0, 0);
    out.src = ctx.toDataURL('image/png');
    var mask = document.querySelector('#output>span.mask');
    mask.style.backgroundImage = `url(${out.src})`;
}

async function save_image(img, name){
    if (! img.src || ! img.naturalWidth || ! img.naturalHeight) return;
    // const opts = {
    //     types: [
    //         {
    //             description: "phantom tank",
    //             accept: {
    //                 // 'image/jpeg': ['.jpg', '.jpeg'],
    //                 'image/png': ['.png']
    //                 // 'image/gif': ['.gif']
    //             }
    //         }
    //     ],
    //     excludeAcceptAllOption: true
    // };
    var data = img.src;
    // var bin = bas64binrary(data);
    // try{
    //     var handle = await window.showSaveFilePicker(opts);
    //     var file = await handle.createWritable();
    //     await file.write(bin[1]);
    //     await file.close();
    // } catch(e) {
    //     console.log(e);
    var a = document.createElement('a');
    a.download = name? name: (Math.random() + '.png').slice(2);
    a.href = data;
    a.click();
    delete a;
}

function color_create(text){
    var c = parseInt(text.value);
    if (isNaN(c) || c < 0 || c > 255){
        text.classList.add('error');
        return;
    }
    var a = parseInt(color0.value);
    var b = parseInt(color1.value);
    if (a >= b){
        color0.classList.add('error');
        color1.classList.add('error');
        return;
    }
    color0.classList.remove('error');
    color1.classList.remove('error');

    out.style.backgroundColor = color_convert.innerText = (color_convert.classList.length == 0)? a: b;
}

function color_revert(){
    var a = parseInt(color0.value), b = parseInt(color1.value),
    c = color_convert.classList.length? b: a, d = c == a? b: a;
    color_convert.innerText = c;
    if(isNaN(a) || isNaN(b) || a < 0 || a >= b || b > 255){
        color_convert.style.backgroundColor = `#444`;
        color_convert.style.color = `#000`;
        return console.log('Color unexcepted.', a, b);
    }
    out.style.backgroundColor = color_convert.style.backgroundColor = `rgb(${c},${c},${c})`; //'#'+c.toString(16).repeat(3);
    color_convert.style.color = `rgb(${d},${d},${d})`;
}

function mouse_move(ctx, img, mask, evt){
    mask.style.left = evt.clientX + 'px';
    mask.style.top = evt.clientY + 'px';
    if (! img.naturalHeight || ! img.naturalWidth) return;
    var scale = img.width / img.naturalWidth,
     x = evt.offsetX/scale, y = evt.offsetY/scale;
    mask.style.backgroundPosition = Math.round(75/scale-x)+'px '+Math.round(75/scale-y)+'px';
//    console.log(x, y, evt.clientX, evt.clientY, scale)
    var pixel = ctx.getImageData(Math.round(x), Math.round(y), 1, 1).data;
    mask.firstChild.innerText = `(${pixel[0]} ${pixel[1]} ${pixel[2]})`;
}

window.onload = () => {
    var mask0 = document.querySelector('#image0>span.mask');
    var mask1 = document.querySelector('#image1>span.mask');
    var mask2 = document.querySelector('#output>span.mask');
     ctx0 = document.querySelector('#image0>canvas').getContext('2d', { willReadFrequently: true });
     ctx1 = document.querySelector('#image1>canvas').getContext('2d', { willReadFrequently: true });
     ctx2 = document.querySelector('#output>canvas').getContext('2d', { willReadFrequently: true });
    mouse_leave = (ctx, img, mask, evt) => {
        mask.style.display = 'none';
    }
    mouse_enter = (ctx, img, mask, evt) => {
        mask.style.display = 'block';
//        mask.style.backgroundImage = `url(${img.src})`;
        mask.style.backgroundColor = color_convert.style.color;     //out.style.backgroundColor;
        var scale = img.naturalWidth? img.width / img.naturalWidth: 1;
        mask.style.fontSize = 10/scale+'px';
        mask.style.transform = `scale(${1.5*scale})`;
        mask.style.width = Math.round(150/scale)+'px';
        mask.style.height = Math.round(150/scale)+'px';
        mouse_move(ctx, img, mask, evt);
    }
    img0.addEventListener("mousemove", (evt) => mouse_move(ctx0, img0, mask0, evt));
    img1.addEventListener("mousemove", (evt) => mouse_move(ctx1, img1, mask1, evt));
    out .addEventListener("mousemove", (evt) => mouse_move(ctx2, out , mask2, evt));
    img0.addEventListener("mouseleave", (evt) => mouse_leave(ctx0, img0, mask0, evt));
    img1.addEventListener("mouseleave", (evt) => mouse_leave(ctx1, img1, mask1, evt));
    out .addEventListener("mouseleave", (evt) => mouse_leave(ctx2, out , mask2, evt));
    img0.addEventListener("mouseenter", (evt) => mouse_enter(ctx0, img0, mask0, evt));
    img1.addEventListener("mouseenter", (evt) => mouse_enter(ctx1, img1, mask1, evt));
    out .addEventListener("mouseenter", (evt) => mouse_enter(ctx2, out , mask2, evt));
    weightSlider.addEventListener('input', function() {
        weightValue.textContent = this.value;
    });
}
// 滑条状态联动
function mode_change(element) {
    weightMode.classList.toggle('on');
    if (element == weightMode){
        color_mode = element.classList.contains('on');
        weightSlider.disabled = ! color_mode;
        weightValue.style.display = weightSlider.style.display = color_mode? 'inline-block': 'none';
    }
//    if(this.checked){
////        cls0.value = 'red green blue';
////        cls1.value = 'red green blue';
//        if(img0.src && img0.naturalWidth && img0.naturalHeight){
//            var ctx0 = document.querySelector('#image0>canvas');
//            img0.src = ctx0.toDataURL('image/png');
//        }
//        if(img1.src && img1.naturalWidth && img1.naturalHeight){
//            var ctx1 = document.querySelector('#image1>canvas');
//            img1.src = ctx1.toDataURL('image/png');
//        }
//    } else {
//        if(img0.src && img0.naturalWidth && img0.naturalHeight) gray(File0);
//        if(img1.src && img1.naturalWidth && img1.naturalHeight) gray(File1);
//    }
}
/**
 * 彩色权重融合（对应 Python phantom_tank_weigth）
 * @param {Uint8ClampedArray} data0 第一张图像的 RGBA 数据（已缩放到公共尺寸）
 * @param {Uint8ClampedArray} data1 第二张图像的 RGBA 数据
 * @param {number} width 公共宽度
 * @param {number} height 公共高度
 * @param {number} bk0 背景色下限 (0-255)
 * @param {number} bk1 背景色上限 (0-255)
 * @param {number} alpha 权重 (0-1)
 * @returns {Uint8ClampedArray} 生成的 RGBA 图像数据
 */
