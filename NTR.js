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

const lookImage = (input) => {
    
    if(input.files.length){
        var file = input.files[0];
        if(file.type.search('image/') == 0){
            var filereader = new FileReader();
            var img = new Image();
            filereader.onload = function(evt){
                if (filereader.readyState == 2){
                    var i = document.querySelector('#' + input.title + '>img');
                    i.src = img.src = evt.target.result;
                    img.onload = function(){
                        var scale = document.querySelector('#' + input.title + '>form>div.alpha').offsetWidth / img.naturalWidth;
                        i.style.width = parseInt(scale * img.width) + 'px';

                        var ctx = document.querySelector('#' + input.title + '>canvas');
                        ctx.height = img.height;
                        ctx.width = img.width;
                        ctx.getContext('2d').drawImage(img, 0, 0);
                    }
                }
            }
            filereader.readAsDataURL(file);
        }
    }
}

const grey = (input) => {
    if (input.files.length == 0)
        return;
    if ((input.title == 'image0' && (! img0.height || ! img0.width))||
        ((input.title == 'image1' && (! img1.height || ! img1.width)))) return console.log('Error image'+input.title.at(-1)+' size.');
    var ctx = document.querySelector('#image'+input.title.at(-1)+'>canvas');
    var data = ctx.getContext('2d').getImageData(0, 0, ctx.width, ctx.height).data;
    var idata = new ImageData(ctx.width, ctx.height);
    var arr = idata.data;
    arr.fill(255);
    var list = (input.title == 'image0')? cls0.classList: cls1.classList;
    if (list.length == 0) return console.log('Error image'+input.title.at(-1)+' color.');
    var li = new Uint8Array(list.length);
    list.forEach((c, i) => {
        if (c == 'red') li[i] = 0;
        else if (c == 'green') li[i] = 1;
        else if (c == 'blue') li[i] = 2;
    });
    for (var i=0, j=data.length, k=0, u=li.length; i < j; i++,k=0){
        li.forEach(c => k += data[i + c]);
        arr[i++] = arr[i++] = arr[i++] = parseInt(k/u);
    }
    var ictx = document.createElement('canvas');
    ictx.width = ctx.width;
    ictx.height = ctx.height;
    ictx.getContext('2d').putImageData(idata, 0, 0);
    if (input.title == 'image0')
        img0.src = ictx.toDataURL('image/png');
    else
        img1.src = ictx.toDataURL('image/png');
    delete ictx;
}

`
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
    assert bk0 < bk1, "If what to get such a different background, Please switch image 0 and image 1"
    assert img0.shape == img1.shape == new.shape[:2]
    arr0 = np.int32(img0.arr)
    arr1 = np.int32(img1.arr)
    arrdx = arr1 - arr0
    c_min = arrdx.min()     # (e1 - e2) / k
    c_max = arrdx.max()     # (b - a) / k + c_min
    d_min = (bk1 * arr0 - bk0 * arr1).min()     # (a * e2 - b * e1) / k
    k = (bk1 - bk0) / (c_max - c_min)
    e1 = k * (d_min + bk0 * c_min) / (bk0 - bk1)
    e2 = k * (d_min + bk1 * c_min) / (bk0 - bk1)
    arr0 = np.int16(np.around(np.multiply(arr0, k, dtype=float) + e1))
    arr1 = np.int16(np.around(np.multiply(arr1, k, dtype=float) + e2))
    print(c_min, c_max, d_min)  #-142 254 -7620
    print(k, e1, e2)
    # print(arr0)
    # print(arr1)
    w = (bk1 - bk0) - (arr1 - arr0)
    q = np.around(np.multiply(w, 255 / (bk1 - bk0), dtype=float))
    # print(q, (q > 255).any(), (q < 0).any())
    "never used np.put !!!!"
    q = q.astype(np.uint8, copy=False)#np.uint8(q)
    zero = w == 0
    w[zero] = 1
    p = np.around((np.multiply(bk1, arr0, dtype=np.int32) - bk0 * arr1) / w)
    np.clip(p, 0, 255, out=p)
    p = np.uint8(p)
    p[zero] = 255

    new[:, :, :3] = p
    new[:, :, 3] = q
    return new

def paste_image(img, background, center=(0, 0)):
    b_c = background.shape[1]//2, background.shape[0]//2
    i_c = img.shape[1]//2, img.shape[0]//2
    pos = b_c[0]+center[0]-i_c[0], b_c[1]+center[1]-i_c[1]
    pad = (max(0, - pos[1]), max(0, pos[1]+img.shape[0]-background.shape[0])),\
          (max(0, - pos[0]), max(0, pos[0]+img.shape[1]-background.shape[1]))
    new = np.pad(background, pad)
    new[max(0, pos[1]): max(0, pos[1])+img.shape[0], max(0, pos[0]): max(0, pos[0])+img.shape[1]] = img
    background[:] = new[pad[0][0]: pad[0][0]+background.shape[0], pad[1][0]: pad[1][0]+background.shape[1]]
`
function phantom_tank(bk, img0, img1, bk0, bk1){
    var arr0 = new Int16Array(img0);
    var arr1 = new Int16Array(img1);
    var w = arr1.map((n, i) => n - arr0.at(i));
    var c_min, c_max;
    c_min = c_max = w[0];
    w.forEach((n) => {
        if (n > c_max) c_max = n;
        else if (n < c_min) c_min = n;
    })
    var d_min;
    // (arr0.map(n => n*bk1) - arr1.map(n => n*bk0))
    d_min = arr0[0]*bk1 - arr1[0]*bk0;
    for (var i=0, j=arr0.length, k; i<j; i++){
        k = arr0[i]*bk1 - arr1[i]*bk0;
        if (k < d_min) d_min = k;
    }
    var dc = bk1 - bk0;
    var k = dc / (c_max - c_min);
    var e1 = k * (d_min + bk0 * c_min) / -dc;
    var e2 = k * (d_min + bk1 * c_min) / -dc;
    for (var i=0, j=arr0.length, u=bk1 - bk0, a, b; i<j; i++){
        arr0[i] = a = Math.round(arr0[i] * k + e1);
        arr1[i] = b = Math.round(arr1[i] * k + e2);
        w[i] = u - (b - a);
    }
    console.log(c_min, c_max, d_min);
    console.log(k, e1, e2);
    k = 255 / dc;
    var q;
    w.forEach((n, i) => {
        q = Math.round(k * n);
        if (q > 255) q = 255;
        else if (q < 0) q = 0;
        bk[i*4+3] = q;
    });
    var p;
    w.forEach((n, i) => {
        if (n == 0) p = 255;
        else {
            p = Math.round((arr0[i] * bk1 - arr1[i] * bk0) / n);
            if (p > 255) p = 255;
            else if (p < 0) p = 0;
        }
        bk[i*=4]=bk[++i]=bk[++i] = p;
    })
}

function paste_image(img, w, bk, w1, center){
    var h = img.length/w;
    var h1 = bk.length/w1;
    var b_c = [parseInt(w1/2), parseInt(h1/2)];
    var i_c = [parseInt(w/2), parseInt(h/2)];
    var pos = [b_c[0]+center[0]-i_c[0], b_c[1]+center[1]-i_c[1]];
    for (var i=0, j, x, y; i<h; i++){
            y = i+pos[1];
        if (0 <= y && y < h1)
            for (j=0; j<w; j++){
            x = j+pos[0];
            if (0 <= x && x < w1)
                bk[y*w1+x] = img[i*w+j];
        }
    }
    console.log(b_c, i_c, pos);
}

function tank_create(){
    if(File0.files.length == 0 || File1.files.length == 0 || ! img0.src || ! img1.src || 
        ! img0.naturalWidth || ! img1.naturalWidth || ! img0.naturalHeight || ! img1.naturalHeight)
        return console.log('onclick: wrroy Image type.');
    var a = parseInt(color0.value);
    var b = parseInt(color1.value);
    if(isNaN(a) || isNaN(b) || a < 0 || a >= b || b > 255)
        return console.log('Color unexcepted.', a, b);
    var size0 = img0.naturalWidth * img0.naturalHeight;
    var size1 = img1.naturalWidth * img1.naturalHeight;
    var bigger = (size0 > size1);
    var scale = Math.sqrt(bigger? size1 / size0: size0 / size1);
    size0 = [bigger? parseInt(img0.naturalWidth*scale): img0.naturalWidth, bigger? parseInt(img0.naturalHeight*scale): img0.naturalHeight];
    size1 = [bigger? img1.naturalWidth: parseInt(img1.naturalWidth*scale), bigger? img1.naturalHeight: parseInt(img1.naturalHeight*scale)];
    var size = [
        Math.max(size0[0], size1[0]), 
        Math.max(size0[1], size1[1])
    ];
    var arr0 = new Uint8Array(size[0] * size[1]).fill((a+b)/2);
    var arr1 = new Uint8Array(size[0] * size[1]).fill((a+b)/2);
    var ctx = document.querySelector("#output>canvas");
    ctx.width = bigger? size0[0]: size1[0];
    ctx.height = bigger? size0[1]: size1[1];
    var c = ctx.getContext('2d');
    c.drawImage(bigger? img0: img1, 0, 0, ctx.width, ctx.height);
    var data = c.getImageData(0, 0, ctx.width, ctx.height);
    var temp = new Uint8Array(ctx.width * ctx.height);
    for (var i=0, j=temp.length; i<j; i++) temp[i] = data.data[i*4];
    paste_image(temp, ctx.width, bigger? arr0: arr1, size[0], [0,0]);
    c.clearRect(0, 0, ctx.width, ctx.height);

    ctx.width = bigger? size1[0]: size0[0];
    ctx.height = bigger? size1[1]: size0[1];
    c.drawImage(bigger? img1: img0, 0, 0);
    data = c.getImageData(0, 0, ctx.width, ctx.height);
    temp = new Uint8Array(size0[0]*size0[1]);
    for (var i=0, j=temp.length; i<j; i++) temp[i] = data.data[i*4];
    paste_image(temp, ctx.width, bigger? arr1: arr0, size[0], [0,0]);
    c.clearRect(0, 0, ctx.width, ctx.height);
    // ctx.width=size[0]
    // ctx.height=size[1]
    // var i=new ImageData(ctx.width, ctx.height)
    // for(var j=0,k=0; j<i.data.length; j++,k++){
    //     i.data[j++]=i.data[j++]=i.data[j++]= arr0[k]
    //     i.data[j] = 255
    // }
    // c.putImageData(i, 0, 0)
    // out.src = ctx.toDataURL('image/png');
    // c.clearRect(0, 0, ctx.width, ctx.height);

    ctx.width = size[0];
    ctx.height = size[1];
    data = c.getImageData(0, 0, ctx.width, ctx.height);
    phantom_tank(data.data, arr0, arr1, a, b);
    c.putImageData(data, 0, 0);
    out.src = ctx.toDataURL('image/png');
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
    var a = parseInt(color0.value);
    var b = parseInt(color1.value);
    color_convert.innerText = (color_convert.classList.length == 0)? a: b;
    if(isNaN(a) || isNaN(b) || a < 0 || a >= b || b > 255){
        color_convert.style.backgroundColor = `#444`;
        color_convert.style.color = `#000`;
        return console.log('Color unexcepted.', a, b);
    }
    var c = (color_convert.classList.length == 0)? a: b;
    out.style.backgroundColor = color_convert.style.backgroundColor = `rgb(${c},${c},${c})`;
    color_convert.style.color = `rgb(${255-c},${255-c},${255-c})`;
}

function mouse_move(ctx, img, mask, evt){
    var x = evt.offsetX;
    var y = evt.offsetY;
    mask.style.left = evt.clientX + 'px';
    mask.style.top = evt.clientY + 'px';
    if (! img.naturalHeight || ! img.naturalWidth) return;
    var scale = img.width / img.naturalWidth;
    mask.style.backgroundPositionX = `-${Math.round((x-75)/scale)}px`;
    mask.style.backgroundPositionY = `-${Math.round((y-75)/scale)}px`;
    // console.log(x, y, evt.clientX, evt.clientY, scale)
    var pixel = ctx.getContext('2d').getImageData(Math.round(x/scale), Math.round(y/scale), 1, 1).data;
    mask.innerText = `(${pixel[0]} ${pixel[1]} ${pixel[2]})`;
}

window.onload = () => {
    var mask0 = document.querySelector('#image0>span.mask');
    var mask1 = document.querySelector('#image1>span.mask');
    var mask2 = document.querySelector('#output>span.mask');
    var ctx0 = document.querySelector('#image0>canvas');
    var ctx1 = document.querySelector('#image1>canvas');
    var ctx2 = document.querySelector('#output>canvas');
    mouse_leave = (ctx, img, mask, evt) => {
        mask.style.display = 'none';
    }
    mouse_enter = (ctx, img, mask, evt) => {
        mask.style.display = 'block';
        mask.style.backgroundImage = `url(${img.src})`;
        var scale = img.naturalWidth? img.width / img.naturalWidth: 1;
        mask.style.fontSize = `${10/scale}px`;
        mask.style.padding = `0 0 ${15/scale}px 0`;
        mask.style.transform = `scale(${1.5*scale})`;
        mask.style.width = `${Math.round(100/scale*1.5)}px`;
        mask.style.height = `${Math.round(100/scale*1.5)}px`;
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
}
