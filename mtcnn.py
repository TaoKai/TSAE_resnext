import os, sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from codecs import open
from PIL import Image, ImageDraw, ImageFont

def bigImgShow(boxes, labels, img):
    for b in boxes:
        b = [int(bb) for bb in b[:4]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255,80,80), 2)
    pilImg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilImg)
    fSize = 20
    font = ImageFont.truetype("simhei.ttf", fSize, encoding="utf-8")
    for l, b in zip(labels, boxes):
        b = [int(bb) for bb in b[:4]]
        top = (b[0], b[1]-fSize-1)
        name = l[0]
        prob = str(l[1])
        if name=='其他':
            name = ''
        prob = ''
        prob = prob[:5] if len(prob)>=5 else prob
        draw.text(top, name+' '+prob, (255, 50, 50), font=font)
    img = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
    return img

def save_pb(sess, names, out_path):
    pb_graph = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=names)
    with tf.gfile.GFile(out_path, "wb") as f:
        f.write(pb_graph.SerializeToString())
        print(names)
        print("%d ops in the final graph." % len(pb_graph.node))

class DetectFace(object):
    def __init__(self):
        self.sess, self.image, self.data, self.fCuts = self.load_pb()

    def getNames(self):
        return ['pnet/input', 'onet/boxes', 'onet/points', 'onet/Ms_inv', 'onet/FinalCuts']
    
    def load_pb(self, path='detect_face.pb'):
        with tf.Graph().as_default():
            config = tf.ConfigProto()  
            config.gpu_options.allow_growth=True  
            sess = tf.Session(config=config)
            pb_graph_def = tf.GraphDef()
            with open(path, "rb") as f:
                pb_graph_def.ParseFromString(f.read())
                tf.import_graph_def(pb_graph_def, name='')
            sess.run(tf.global_variables_initializer())
            image = sess.graph.get_tensor_by_name("pnet/input:0")
            boxes = sess.graph.get_tensor_by_name("onet/boxes:0")
            points = sess.graph.get_tensor_by_name("onet/points:0")
            Ms = sess.graph.get_tensor_by_name("onet/Ms_inv:0")
            fCuts = sess.graph.get_tensor_by_name("onet/FinalCuts:0")
            return sess, image, [boxes, points, Ms], fCuts
    
    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cuts, data = self.sess.run([self.fCuts, self.data], {self.image:img})
        return cuts, data

def drawAll(data, results, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = data[0]
    labels = results
    bigImg = bigImgShow(boxes, labels, img)
    return bigImg

def cutByBoxes(img, boxes):
    boxes = list(boxes.astype(np.int32))
    cuts = []
    for b in boxes:
        w = b[2]-b[0]
        h = b[3]-b[1]
        new_w = int(h/112*96)
        new_x = int(b[0]+(w-new_w)/2)
        new_x = 0 if new_x<0 else new_x
        cut = img[b[1]:b[3], new_x:new_x+new_w, :]
        orig_cut = img[b[1]:b[3], b[0]:b[2], :]
        cut = cv2.resize(cut, (96, 112), interpolation=cv2.INTER_LINEAR)
        cuts.append(cut)
    return np.array(cuts, dtype=np.float32)

def border_filter(data, res):
    for i, b in enumerate(data[0]):
        w = b[2]-b[0]
        h = b[3]-b[1]
        if w<20 or h<25:
            res[i][0] = '其他'

def test():
    detectFace = DetectFace()
    path = 'videos/andy.mp4'
    cap = cv2.VideoCapture(path)
    start = 2000
    intv = 6
    total = 5000
    cnt = 0
    pic_cnt = 0
    name = path.split('/')[-1].split('.')[0]
    facePath = 'faces/'+name
    if not os.path.exists(facePath):
        os.makedirs(facePath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cnt>=start and cnt%intv==0:
                cuts, _ = detectFace(frame)
                if cuts is None or cuts.shape[0]==0:
                    continue
                if cuts.shape[0]>2:
                    continue
                cut = cuts[0]
                cut = cv2.cvtColor(cut, cv2.COLOR_RGB2BGR)
                fp = facePath+'/'+name+'_'+str(pic_cnt)+'.jpg'
                cv2.imwrite(fp, cut)
                pic_cnt += 1
                if pic_cnt>total:
                    break
                # cv2.imshow('cut', cut)
                # cv2.waitKey(1)
            print(cnt, pic_cnt)
            cnt += 1
        else:
            break
    cap.release()
    
    

if __name__ == "__main__":
    test()