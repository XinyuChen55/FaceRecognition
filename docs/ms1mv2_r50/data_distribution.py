import mxnet
from collections import Counter
from mxnet import recordio
import numpy as np

rec_path = "/home/chelsea/FaceRecognition/data/MS1M/train.rec"
idx_path = "/home/chelsea/FaceRecognition/data/MS1M/train.idx"
subset_path = "/home/chelsea/FaceRecognition/third_party/insightface/recognition/arcface_torch/subset_imgidx.npy"
subset_imgidx = np.load(subset_path)

imgrec = recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

s = imgrec.read_idx(0)
header, _ = recordio.unpack(s)

print("header0 label:", header.label)
print("header0 flag:", header.flag)

max_img_idx = int(header.label[0])
counter = Counter()

for i in subset_imgidx:
    s = imgrec.read_idx(i)
    header, _ = recordio.unpack(s)
    label = header.label
    identity = int(label)
    counter[identity] += 1

freq = np.array(list(counter.values()))

print("num identities:", len(freq))
print("num images counted:", freq.sum())
print("mean:", freq.mean())
print("median:", np.median(freq))
print("std:", freq.std())
print("min:", freq.min())
print("max:", freq.max())
