# Week 3 进度报告

1. 本周我完成了任务三，学习了MTCNN和RetinaFace的人脸检测算法原理。

2. 我还用MMDetection的Retinanet框架在WIDER FACE上训练了一个人脸检测模型。
在下载了WIDER FACE数据集以后，我写了代码 [wider_to_coco.py](../docs/wider_to_coco.py) 把WIDER FACE的 ground truth table 转成了 COCO json 格式，然后稍微修改了MMDetection自带的配置文件 retinanet_r50_fpn_1x.py => [retinanet_r50_fpn_1x_widerface.py](../configs/retinanet_r50_fpn_1x_widerface.py)，让它只检测face一个种类并且把训练的数据集改成WIDER FACE。训练完成后的评估报告在 [widerface_eval_w3.md](./widerface_evalw3.md)。

3. 最后我用训练得到的权重文件 [checkpoints](../work_dirs/retinanet_r50_fpn_1x_widerface/latest.pth) 从WIDER FACE的测试图像里随机抽了十张图像出来进行检测，检测结果在 [测试结果文档](../assets/outputs/widerface_test)。

现在我已经开始了任务四的进度，学习了HRNet，SAN的人脸关键点检测的常用算法，并下载了300W数据集。