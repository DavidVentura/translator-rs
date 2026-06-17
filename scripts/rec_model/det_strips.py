"""Run text detection on each image, save dewarped strips + a numbered sheet.
Usage: det_strips.py <det.mnn> <image_dir>
Outputs: <image_dir>/strips/<image>-<NN>.jpg  and  <image>_sheet.png (for VLM reading)."""
import sys, glob, os
import numpy as np, cv2, MNN
from PIL import Image, ImageDraw, ImageFont
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DET, IMGDIR = sys.argv[1], sys.argv[2]

itp = MNN.Interpreter(DET); sess = itp.createSession(); inp = itp.getSessionInput(sess)
def infer(x):
    shp = x.shape; itp.resizeTensor(inp, shp); itp.resizeSession(sess)
    inp.copyFrom(MNN.Tensor(shp, MNN.Halide_Type_Float, np.ascontiguousarray(x), MNN.Tensor_DimensionType_Caffe))
    itp.runSession(sess); o = itp.getSessionOutput(sess); osh = tuple(o.getShape())
    ot = MNN.Tensor(osh, MNN.Halide_Type_Float, np.zeros(osh,np.float32), MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ot); return np.array(ot.getData()).reshape(osh)
def detect(rgb):
    h0,w0 = rgb.shape[:2]; sc = min(960/max(h0,w0),1.0)
    nh = max(32,int(round(h0*sc/32))*32); nw = max(32,int(round(w0*sc/32))*32)
    im = cv2.resize(rgb,(nw,nh)).astype(np.float32)/255.0
    prob = infer(np.transpose((im-MEAN)/STD,(2,0,1))[None]).squeeze()
    ph,pw = prob.shape; sx,sy = w0/pw, h0/ph
    binmap = cv2.dilate((prob>0.3).astype(np.uint8), np.ones((3,3),np.uint8), 2)
    cnts,_ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        x,y,bw,bh = cv2.boundingRect(c)
        if bw*bh<64 or prob[y:y+bh,x:x+bw].mean()<0.6: continue
        (cx,cy),(rw,rh),ang = cv2.minAreaRect(c)
        cx*=sx; cy*=sy; rw*=sx; rh*=sy
        D = rw*rh*1.0/(2*(rw+rh)+1e-6)        # DBNet unclip distance (ratio=1.0)
        b = cv2.boxPoints(((cx,cy),(rw+2*D+8,rh+2*D+8),ang)); boxes.append(b)
    return boxes
def order(p):
    p = p[np.argsort(p[:,1])]; t = p[:2][np.argsort(p[:2,0])]; b = p[2:][np.argsort(p[2:,0])]
    return np.array([t[0],t[1],b[1],b[0]], np.float32)
def cropbox(rgb, box):
    b = order(box.astype(np.float32))
    W = int(max(np.linalg.norm(b[0]-b[1]), np.linalg.norm(b[3]-b[2])))
    H = int(max(np.linalg.norm(b[0]-b[3]), np.linalg.norm(b[1]-b[2])))
    if W<10 or H<8: return None
    M = cv2.getPerspectiveTransform(b, np.array([[0,0],[W,0],[W,H],[0,H]],np.float32))
    c = cv2.warpPerspective(rgb, M, (W,H))
    return cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE) if H>W*1.5 else c

sdir = os.path.join(IMGDIR,"strips"); os.makedirs(sdir, exist_ok=True)
imgs = sorted(f for f in glob.glob(IMGDIR+"/*") if f.lower().endswith((".jpg",".png")) and "_pred" not in f)
font = ImageFont.load_default(size=22)
for path in imgs:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    name = os.path.splitext(os.path.basename(path))[0]
    items = []
    for b in detect(rgb):
        c = cropbox(rgb,b)
        if c is not None: items.append((order(b)[0][1], order(b)[0][0], c))
    items.sort(key=lambda s:(round(s[0]/20), s[1]))
    cells = []
    for i,(_,_,c) in enumerate(items):
        cv2.imwrite(os.path.join(sdir,f"{name}-{i:02d}.jpg"), c[:,:,::-1])
        sh = Image.fromarray(c); sh = sh.resize((min(620, max(16,sh.width*48//max(1,sh.height))),48))
        cell = Image.new("RGB",(680,56),(255,255,255)); d = ImageDraw.Draw(cell)
        d.text((4,16),f"{i:02d}",fill=(200,0,0),font=font); cell.paste(sh,(54,4)); cells.append(np.asarray(cell))
    if cells: Image.fromarray(np.concatenate(cells,0)).save(os.path.join(sdir,f"{name}_sheet.png"))
    print(f"{name}: {len(items)} strips -> {sdir}/{name}-NN.jpg")
