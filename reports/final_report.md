# 字节跳动智能视觉AI：人脸识别与特效技术结项报告

## 项目成果总结

1. 首先在项目初期，我完成了基础开发环境的搭建，包括 Python、PyTorch、OpenCV 以及人脸相关的依赖库 MMDetection、MMHuman3D 等 ，同时学习了 Git 和 Docker 的基本使用方法。完成环境验证后，我编写并运行了基础 OpenCV 图像处理程序，比如图像读取、灰度化和显示等功能

2. 然后我学习了人脸检测、人脸关键点定位、人脸对齐、人脸验证与人脸识别等基本概念，并对 CelebA、LFW 等人脸数据集进行了观察与可视化分析。我还用MMDetection自带的人脸检测模型在几张图上进行了人脸检测，部分检测结果如下。我也初步建立了对人脸识别完整流程的理解：先进行人脸检测，再通过关键点实现对齐，随后提取人脸特征并进行身份验证或识别。
    <p align="center">
    <img src="../assets/outputs/mmdet_test/vis/Carlos_Salinas_0001.jpg" alt="Result 1" width="30%" />
    <img src="../assets/outputs/mmdet_test/vis/Recep_Tayyip_Erdogan_0003.jpg" alt="Result 2" width="30%" />
    <img src="../assets/outputs/mmdet_test/vis/Sandra_Milo_0001.jpg" alt="Result 3" width="30%" />
    </p>

3.  在人脸检测部分，我主要学习了MTCNN、RetinaFace 等算法的基本原理，并MMDectection的框架在WIDER FACE数据集上训练了一个人脸检测模型，部分检测结果如下。可以看出训练的模型能够在图像上较稳定地检测出人脸区域。这阶段也让我更加清楚地理解了多尺度检测、召回率和精度等的概念。
    <p align="center">
    <img src="../assets/outputs/widerface_test/14--Traffic/14_Traffic_Traffic_14_855.jpg" alt="Image 1" width="48%" />
    <img src="../assets/outputs/widerface_test/19--Couple/19_Couple_Couple_19_282.jpg" alt="Image 2" width="48%" />
    </p>

    <p align="center">
    <img src="../assets/outputs/widerface_test/35--Basketball/35_Basketball_basketballgame_ball_35_357.jpg" alt="Image 3" width="48%" />
    <img src="../assets/outputs/widerface_test/41--Swimming/41_Swimming_Swimming_41_712.jpg" alt="Image 4" width="48%" />
    </p>

4. 在人脸关键点检测与对齐部分，我进一步学习了 HRNet 和 SAN 等关键点检测算法，并用MMPose里的HRNet配置文件在300W数据集上进行了训练与测试，部分关键点检测结果如下。同时，基于检测到的人脸关键点，我也实现了人脸对齐，通过仿射变换将人脸的几何位置进行统一，从而减少后续识别任务的干扰。
    <div>人脸关键点检测结果</div>
    <p align="center">
    <img src="../assets/outputs/300w_keypoint_test/indoor_001_0.png" alt="Image 1" width="48%" />
    <img src="../assets/outputs/300w_keypoint_test/indoor_002_0.png" alt="Image 2" width="48%" />
    </p>

    <p align="center">
    <img src="../assets/outputs/300w_keypoint_test/outdoor_001_0.png" alt="Image 3" width="48%" />
    <img src="../assets/outputs/300w_keypoint_test/outdoor_004_0.png" alt="Image 4" width="48%" />
    </p>

    <div>对应的人脸对齐结果</div>
    <p align="center">
    <img src="../assets/outputs/300w_keypoint_test/face_aligned/indoor_001_aligned.png" alt="Image 1" width="24%" />
    <img src="../assets/outputs/300w_keypoint_test/face_aligned/indoor_002_aligned.png" alt="Image 2" width="24%" />
    <img src="../assets/outputs/300w_keypoint_test/face_aligned/outdoor_001_aligned.png" alt="Image 3" width="24%" />
    <img src="../assets/outputs/300w_keypoint_test/outdoor_004_0.png" alt="Image 4" width="24%" />
    </p>

5. 在人脸识别模型训练部分，我学习了基于 ResNet 和 ArcFace loss 的人脸识别方法。然后用InsightFace 里的 ResNet-50 配置文件在 MS-Celeb-1M 的子集上完成了训练实验，训练过程中的损失曲线和准确率曲线如下。我从 MS-Celeb-1M 里用抽取不放回的方式随机抽取了五十万张照片作为子集，然后考虑到有些 identity 只有 1–2 张图像的极端情况会影响模型的泛化，我直接去除了这部分 identity 提高了数据的合理性。训练完成后我在 LFW 数据集上进行了人脸验证，准确率高达99.47%。通过这些任务我对 数据集合理性、embedding、特征归一化、余弦相似度以及人脸验证的基本机制有了更加深入的理解。
    <p align="center">
    <img src="./loss_curve.png" alt="Image 3" width="48%" />
    <img src="./train_acc_curve.png" alt="Image 4" width="48%" />
    </p>

6. 在模型优化与部署准备部分，我学习了模型量化、剪枝和蒸馏等优化技术，并用PyTorch提供的工具实现了动态量化与 ONNX 导出。我先将训练好的模型切换到推理模式，再对部分层进行动态量化，以减小模型体积并提升 CPU 端推理效率。之后，我将模型导出为 ONNX 格式，并使用 ONNX Runtime 进行推理测试。然后我进行了对比，原PyTorch模型的输出和量化后的ONNX模型的输出的最大绝对误差是0.00000176，平均绝对误差是0.00000045，误差比较小，说明模型成功量化并转换为ONNX格式，且能进行ONNX推理。我也意识到一个模型不仅需要具备较好的训练效果，也需要考虑对其优化和部署的效果。

7. 在人脸特效算法实现部分，我首先学习了 GAN 的基本原理，并了解了 StarGAN 和 AttGAN 等多属性编辑的模型。同时我也用 StarGAN 模型在 CelebA 数据机上进行了训练，主要选择了六个属性进行编辑，分别是黑发，金发，棕发，性别，年龄和笑容。训练结束后，我也进行了测试，部分测试结果如下。
    <p align="center">
    <img src="../assets/outputs/attrs_edit/1-images.jpg" alt="Image 4" width="55%" />
    </p>

8. 在 3D 人脸重建任务中，我学习了 3DMM、NeRF 等三维重建的基础原理，并使用 3DDFA_V2 从单张图像恢复 3D 人脸结构，然后将得到的 .obj 文件导入 MeshLab 进行可视化，部分的重建结果如下。之后，我利用 PyTorch3D 对重建结果进行渲染和可视化，并借助多角度观察分析模型效果，部分结果如下。从整体结果来看，正面和小角度侧脸的重建效果相对较好，而对于大角度侧脸和不可见区域容易出现纹理拉伸和细节失真。

    <p align="center">
    <img src="../assets/test_imgs/3d_reconstruct/1602308_1.jpg" alt="Image 2" width="30%" />
    <img src="./image-8.png" alt="Image 2" width="30%" />
    <img src="../assets/outputs/3d_reconstruct_render/1602308_1_render.png" alt="Image 2" width="30%" />
    </p>

    <p align="center">
    <img src="../assets/test_imgs/3d_reconstruct/16413031_1.jpg" alt="Image 3" width="30%" />
    <img src="./image-9.png" alt="Image 3" width="30%" />
    <img src="../assets/outputs/3d_reconstruct_render/16413031_1_render.png" alt="Image 4" width="30%" />
    </p>

    <div>对第一张图的渲染结果进行的多角度观察</div>
    <p align="center">
    <img src="../assets/outputs/3d_reconstruct_multi_view/front.png" alt="Image 3" width="30%" />
    <img src="../assets/outputs/3d_reconstruct_multi_view/left30.png" alt="Image 3" width="30%" />
    <img src="../assets/outputs/3d_reconstruct_multi_view/left60.png" alt="Image 4" width="30%" />
    </p>

    <p align="center">
    <img src="../assets/outputs/3d_reconstruct_multi_view/top.png" alt="Image 3" width="24%" />
    <img src="../assets/outputs/3d_reconstruct_multi_view/down.png" alt="Image 3" width="24%" />
    <img src="../assets/outputs/3d_reconstruct_multi_view/right30.png" alt="Image 3" width="24%" />
    <img src="../assets/outputs/3d_reconstruct_multi_view/right60.png" alt="Image 4" width="24%" />
    </p>

9. 在动态特效实现部分，我完成了基于实时人脸关键点检测的动态贴纸和基础美颜美妆功能。首先我了解了怎么用OpenCV读取摄像头画面，并用MediaPipe对图像进行人脸关键点检测，然后把结果实时返回到摄像头画面里，这样视频里就能实现实时的人脸关键点检测，然后根据具体的关键点位置，我将眼镜和帽子贴纸叠加到检测到的人脸区域上。同时，我还实现了磨皮、美白和口红三种美颜特效。主要通过构造人脸区域 mask，再将处理后的图像与原图进行局部融合来完成。通过这些任务，我将前面学习到的人脸检测、关键点与图像处理能力一起应用到了实时视频场景中。部分的特效实现结果如下，更多测试的结果在 [weekly_report_w8.md](./weekly_report_w8.md)。

    <div>动态眼镜贴纸特效和歪头的效果</div>
    <p align="center">
    <img src="./effects_results/image-8.png" alt="Image 3" width="48%" />
    <img src="./effects_results/image-9.png" alt="Image 4" width="48%" />
    </p>

    <div>原图以及磨皮、美白、口红的美颜特效效果</div>
    <p align="center">
    <img src="./effects_results/image-12.png" alt="Image 3" width="45%" />
    <img src="./effects_results/image-13.png" alt="Image 4" width="45%" />
    </p>
    <p align="center">
    <img src="./effects_results/image-14.png" alt="Image 3" width="45%" />
    <img src="./effects_results/image-15.png" alt="Image 4" width="45%" />
    </p>