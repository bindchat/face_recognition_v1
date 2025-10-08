# YOLO 人脸识别系统

[English](README.md) | [中文](README.zh-CN.md)

一个综合性的人脸识别系统，使用 YOLO 进行人脸检测，并通过深度学习进行人脸识别。支持图像和实时摄像头识别，配有易于使用的人脸数据库管理工具。

## 功能特性

- 🎯 **基于 YOLO 的人脸检测** - 使用 YOLOv8 进行快速准确的人脸检测
- 👤 **人脸识别** - 从数据库中识别已知人脸
- 📸 **图像识别** - 处理单张图像
- 📹 **摄像头识别** - 从网络摄像头/摄像头进行实时识别
- 🗄️ **数据库管理** - 易于使用的人脸数据库管理工具
- 🎨 **可视化反馈** - 带有置信度分数的边界框和标签

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意：** 安装 `dlib` 可能需要额外的系统依赖：
- **Ubuntu/Debian:** `sudo apt-get install cmake libopenblas-dev liblapack-dev`
- **macOS:** `brew install cmake`
- **Windows:** 建议使用预编译的 wheel 包，可从[这里](https://github.com/jloh02/dlib/releases)获取

### 2. 下载 YOLO 模型（首次运行时自动下载）

YOLOv8 模型将在首次使用时自动下载。

## 快速开始

### 步骤 1：创建人脸数据库

按照以下目录结构组织您的人脸图像：

```
faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── person3/
    └── photo1.jpg
```

然后将人脸导入数据库：

```bash
python face_database.py import faces/
```

### 步骤 2：识别人脸

**从图像识别：**
```bash
python recognize_image.py photo.jpg
```

**从摄像头识别（实时）：**
```bash
python recognize_camera.py
```

按 `q` 键退出摄像头视图。

## 使用指南

### 人脸数据库管理

`face_database.py` 脚本用于管理人脸数据库。

#### 从目录导入人脸：
```bash
python face_database.py import <目录>
```

#### 添加单个人脸：
```bash
python face_database.py add <图像路径> <人名>
```

示例：
```bash
python face_database.py add john_photo.jpg "John Doe"
```

#### 列出数据库中的所有人脸：
```bash
python face_database.py list
```

#### 清空数据库：
```bash
python face_database.py clear
```

#### 使用自定义数据库文件：
```bash
python face_database.py --db my_faces.pkl import faces/
```

### 图像识别

使用 `recognize_image.py` 处理单张图像文件：

#### 基本用法：
```bash
python recognize_image.py photo.jpg
```

#### 保存输出到文件：
```bash
python recognize_image.py photo.jpg --output result.jpg
```

#### 不显示结果窗口：
```bash
python recognize_image.py photo.jpg --output result.jpg --no-show
```

#### 使用自定义数据库和置信度：
```bash
python recognize_image.py photo.jpg --db my_faces.pkl --confidence 0.6
```

### 摄像头识别

使用 `recognize_camera.py` 从摄像头进行实时人脸识别：

#### 基本用法（默认摄像头）：
```bash
python recognize_camera.py
```

#### 使用不同的摄像头：
```bash
python recognize_camera.py --camera-id 1
```

#### 使用自定义数据库和设置：
```bash
python recognize_camera.py --db my_faces.pkl --confidence 0.6
```

**控制方式：**
- 按 `q` 键退出

### 高级用法

#### 直接使用 Python API：

```python
from face_recognition_yolo import YOLOFaceRecognizer

# 初始化识别器
recognizer = YOLOFaceRecognizer(
    db_path='face_database.pkl',
    yolo_model='yolov8n.pt',
    confidence=0.5
)

# 处理图像
results = recognizer.process_image('photo.jpg', output_path='result.jpg')

# 启动摄像头识别
recognizer.process_camera(camera_id=0)
```

#### 以编程方式管理人脸数据库：

```python
from face_database import FaceDatabase

# 创建/加载数据库
db = FaceDatabase('face_database.pkl')

# 添加人脸
db.add_face_from_image('photo.jpg', 'John Doe')

# 从目录导入
db.import_from_directory('faces/')

# 列出人脸
db.list_faces()

# 保存数据库
db.save_database()
```

## 配置选项

### YOLO 模型

您可以使用不同的 YOLO 模型来平衡速度和准确性：

- `yolov8n.pt` - Nano（最快，默认）
- `yolov8s.pt` - Small（小型）
- `yolov8m.pt` - Medium（中型）
- `yolov8l.pt` - Large（大型）
- `yolov8x.pt` - Extra Large（超大型，最准确）

示例：
```bash
python recognize_camera.py --model yolov8m.pt
```

### 置信度阈值

调整检测置信度阈值（默认：0.5）：

```bash
python recognize_image.py photo.jpg --confidence 0.7
```

较低的值 = 更多检测结果（可能包含误报）
较高的值 = 更少检测结果（可能遗漏某些人脸）

## 工作原理

1. **人脸检测：** YOLO 检测图像/帧中的人脸
2. **人脸编码：** 每个检测到的人脸被转换为 128 维编码
3. **人脸匹配：** 将编码与数据库中的已知人脸进行比较
4. **识别：** 如果找到匹配（距离 < 0.6），则用该人的姓名标记人脸

## 故障排除

### 未检测到人脸
- 确保人脸清晰可见且不太小
- 尝试降低 `--confidence` 阈值
- 检查图像质量和光照

### 识别准确度较差
- 在数据库中为每个人添加更多照片（建议 3-5 张）
- 使用不同角度和光照的照片
- 确保照片清晰且人脸正面朝向

### 摄像头无法打开
- 检查摄像头权限
- 尝试不同的 `--camera-id` 值（0、1、2 等）
- 确保没有其他应用程序正在使用摄像头

### dlib 安装问题
- Ubuntu 系统：`sudo apt-get install cmake libopenblas-dev liblapack-dev`
- macOS 系统：`brew install cmake`
- Windows 系统建议使用预编译的 wheel 包

## 项目结构

```
.
├── face_database.py          # 人脸数据库管理工具
├── face_recognition_yolo.py  # 主要人脸识别模块
├── recognize_image.py        # 图像识别脚本
├── recognize_camera.py       # 摄像头识别脚本
├── requirements.txt          # Python 依赖
├── face_database.pkl         # 人脸数据库（生成的）
└── README.md                 # 说明文件
```

## 系统要求

- Python 3.7+
- OpenCV
- YOLOv8 (ultralytics)
- face_recognition
- dlib
- numpy
- pillow

## 在 Jetson Orin Nano 上使用 GPU 加速

以下说明帮助你在 NVIDIA Jetson Orin Nano 开发者套件上启用 GPU 推理。

### 1. 基础环境准备（JetPack）

- 建议使用预装 JetPack 的官方系统镜像（包含 CUDA/cuDNN/TensorRT）。
- 更新系统依赖：
```bash
sudo apt update && sudo apt install -y python3-pip python3-dev cmake libopenblas-dev liblapack-dev
```

### 2. 安装 Python 依赖

Jetson 上安装 `dlib` 和 `opencv-python` 可能较慢，确保网络通畅：
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

> 提示：如遇 `dlib` 编译困难，可尝试预编译 wheel 或参考官方仓库的安装建议。

### 3. 运行时启用 GPU/半精度

本项目已支持选择推理设备与半精度 FP16。默认将自动检测 CUDA 并在可用时启用半精度（更省显存，Jetson 友好）。

- 图像识别（GPU，自动设备+FP16）：
```bash
python3 recognize_image.py photo.jpg --device 0
```

- 摄像头识别（指定 GPU0 并启用 FP16）：
```bash
python3 recognize_camera.py --device 0
```

- 如遇半精度不兼容或显示异常，可禁用 FP16：
```bash
python3 recognize_camera.py --device 0 --no-half
```

### 4. 首帧加速（GPU 预热）

程序会在加载模型后进行一次轻量 GPU 预热，以减少首帧延迟；如失败会自动忽略，不影响正常运行。

### 5. 常见问题（Jetson）

- 启动慢或首帧卡顿：首次运行需要下载 YOLO 模型且触发 CUDA/Tensor 核编译，属正常现象；后续运行会更快。
- 显存不足：使用较小的模型（如 `yolov8n.pt`），或添加 `--no-half`/降低分辨率。
- 摄像头无法打开：检查 `/dev/video*` 权限和设备号，尝试 `--camera-id 0/1/2`。

## 许可证

本项目是开源的，采用 MIT 许可证。

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 用于人脸检测
- [face_recognition](https://github.com/ageitgey/face_recognition) 用于人脸编码和匹配
- [OpenCV](https://opencv.org/) 用于图像处理