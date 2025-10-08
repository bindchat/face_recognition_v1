#!/usr/bin/env python3
"""
【YOLO人脸识别模块】
这是核心识别程序，结合了两种技术：
1. YOLO - 用来在图片中快速找到人脸的位置（就像在照片中圈出脸）
2. face_recognition - 用来识别圈出来的脸是谁（就像认出朋友的脸）
"""

# 导入需要的工具包
import cv2  # OpenCV - 图像处理工具，可以读取和处理图片
import numpy as np  # NumPy - 数学计算工具，用来处理数字和数组
import face_recognition  # 人脸识别工具
import pickle  # 用来读取保存的数据
import os  # 操作系统工具
from ultralytics import YOLO  # YOLO模型 - 快速物体检测工具
try:
    import torch  # 可选：用于检测CUDA与半精度可用性
except Exception:
    torch = None
from PIL import Image, ImageDraw, ImageFont  # 使用Pillow渲染中文文字
from typing import Optional, List, Dict, Union


class YOLOFaceRecognizer:
    """
    【YOLO人脸识别器类】
    这是一个智能的人脸识别器，可以：
    1. 在图片或视频中找到人脸
    2. 识别出每张脸是谁
    3. 在脸上画框并标注名字
    """
    
    def __init__(self, db_path='face_database.pkl', yolo_model='yolov8n.pt', confidence=0.5,
                 font_path: Optional[str] = None, font_size: int = 20,
                 device: Optional[Union[str, int]] = None, use_half: Optional[bool] = None):
        """
        【初始化识别器】
        准备好识别器需要的所有工具和数据
        
        参数说明：
            db_path: 人脸数据库文件的位置
            yolo_model: YOLO模型文件（用来检测人脸位置）
            confidence: 信心阈值（0到1之间，越高要求越严格）
                       例如0.5表示至少有50%把握才认为找到了人脸
        """
        self.db_path = db_path  # 保存数据库路径
        self.confidence = confidence  # 保存信心阈值
        self.known_face_encodings = []  # 存放已知人脸的特征（空列表）
        self.known_face_names = []  # 存放已知人脸的名字（空列表）
        # 中文文字渲染配置
        self.font_path = font_path  # 可选：自定义中文字体路径
        self.font_size = font_size  # 可选：中文标签字号
        self._font = None  # 延迟加载字体
        # 推理设备/精度配置（为 Jetson GPU 友好）
        self.yolo_device = self._resolve_device(device)
        self.yolo_half = self._resolve_half_precision(use_half)
        
        # 加载人脸数据库
        self.load_database()
        
        # 初始化YOLO模型
        print("正在加载YOLO模型...")
        self.yolo_model = YOLO(yolo_model)  # 加载AI模型
        print(f" YOLO模型加载完成（device={self.yolo_device or 'auto'}, half={self.yolo_half}）")
        # 轻量化 GPU 预热，降低首帧延迟
        self._warmup_model()

    def _resolve_device(self, device_hint: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
        # 显式指定优先
        if device_hint is not None and device_hint != '':
            return device_hint
        # 自动：如果可用则用 CUDA:0，否则 CPU
        try:
            if torch is not None and torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return 'cpu'

    def _resolve_half_precision(self, use_half_hint: Optional[bool]) -> bool:
        if isinstance(use_half_hint, bool):
            return use_half_hint
        # 自动：仅在 CUDA 上默认使用半精度（Jetson 友好，节省显存）
        try:
            return bool(torch and torch.cuda.is_available())
        except Exception:
            return False

    def _warmup_model(self) -> None:
        try:
            # 仅在 CUDA 上进行一次小分辨率的空跑，触发内核编译/权重搬运
            if self.yolo_device == 'cpu' or self.yolo_device == 'mps':
                return
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.yolo_model(dummy, verbose=False, device=self.yolo_device, half=self.yolo_half)
        except Exception:
            # 预热失败不应影响后续功能
            pass

    def _resolve_font_path(self) -> Optional[str]:
        """
        解析中文字体路径：优先使用传入的 font_path 或环境变量，其次尝试常见系统路径。
        返回可用字体文件的绝对路径，如果找不到则返回 None。
        """
        # 优先级：构造参数 > 环境变量 > 常见路径
        candidate_paths: list[str | None] = [
            self.font_path,
            os.getenv('CHINESE_FONT_PATH'),
            # Linux 常见中文字体
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',           # 文泉驿正黑
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',         # 文泉驿微米黑
            '/usr/share/fonts/truetype/arphic/ukai.ttc',              # AR PL UKai
            # macOS 常见中文字体
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            # Windows 常见中文字体
            r'C:\\Windows\\Fonts\\msyh.ttc',
            r'C:\\Windows\\Fonts\\msyhbd.ttc',
            r'C:\\Windows\\Fonts\\simhei.ttf',
            r'C:\\Windows\\Fonts\\simsun.ttc',
        ]

        for path in candidate_paths:
            if path and os.path.exists(path):
                return path
        return None

    def _get_font(self) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        """
        懒加载字体对象。如果找到中文字体则返回 TrueType 字体，找不到则回退默认字体。
        """
        if self._font is not None:
            return self._font

        resolved_path = self._resolve_font_path()
        try:
            if resolved_path:
                self._font = ImageFont.truetype(resolved_path, self.font_size)
            else:
                # 回退到默认字体（可能不支持中文，但避免崩溃）
                self._font = ImageFont.load_default()
        except Exception:
            self._font = ImageFont.load_default()
        return self._font

    def _draw_labels_pil(self, frame: np.ndarray, labels_info: List[Dict]) -> np.ndarray:
        """
        使用 Pillow 在图像上绘制中文标签，避免 OpenCV putText 的中文乱码问题。

        labels_info: 每项包含 {text, x1, y1, x2, y2, bg_bgr}
        返回：绘制完标签的 BGR 图像
        """
        if not labels_info:
            return frame

        # OpenCV(BGR ndarray) -> PIL(RGB Image)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        drawer = ImageDraw.Draw(pil_image)
        font = self._get_font()

        image_height, image_width = frame.shape[:2]

        for info in labels_info:
            text = info['text']
            x1 = info['x1']
            y1 = info['y1']
            x2 = info['x2']
            y2 = info['y2']
            bg_bgr = info['bg_bgr']

            # 计算文本尺寸
            try:
                left, top, right, bottom = drawer.textbbox((0, 0), text, font=font)
                text_w = max(1, right - left)
                text_h = max(1, bottom - top)
            except Exception:
                # 旧 Pillow 兜底
                text_w, text_h = drawer.textsize(text, font=font)

            # 计算放置位置：优先放在框上方，放不下则放在框下方
            tx = max(0, min(x1, image_width - text_w - 8))
            ty = y1 - text_h - 8
            if ty < 0:
                ty = min(y2 + 2, max(0, image_height - text_h - 6))

            # 背景矩形
            box_x1 = tx
            box_y1 = ty
            box_x2 = min(image_width - 1, tx + text_w + 8)
            box_y2 = min(image_height - 1, ty + text_h + 6)

            # 颜色转换 BGR -> RGB
            bg_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))

            drawer.rectangle([(box_x1, box_y1), (box_x2, box_y2)], fill=bg_rgb)
            drawer.text((tx + 4, ty + 3), text, fill=(255, 255, 255), font=font)

        # PIL(RGB) -> OpenCV(BGR)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def load_database(self):
        """
        【加载人脸数据库】
        从文件中读取之前保存的人脸数据
        就像打开通讯录，查看里面保存的所有联系人
        
        返回值：
            True 表示成功，False 表示失败
        """
        # 检查数据库文件是否存在
        if not os.path.exists(self.db_path):
            print(f" 警告：找不到人脸数据库文件：{self.db_path}")
            print("  请先运行 'python face_database.py import <文件夹>' 来创建数据库")
            return False
        
        try:
            # 打开并读取数据库文件
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)  # 读取保存的数据
                self.known_face_encodings = data.get('encodings', [])  # 获取人脸特征
                self.known_face_names = data.get('names', [])  # 获取人脸名字
            
            print(f" 成功加载了 {len(self.known_face_names)} 张人脸数据")
            return True
        except Exception as e:
            print(f" 加载数据库时出错：{e}")
            return False
    
    def recognize_faces_in_frame(self, frame):
        """
        【在一帧图像中检测和识别人脸】
        这是核心功能！分为两步：
        1. 用YOLO找出图片中所有人脸的位置
        2. 识别每张脸是谁
        
        参数说明：
            frame: 一张图片（BGR格式，来自OpenCV）
            
        返回值：
            结果列表，每个结果包含：(名字, 信心值, 位置框)
            例如：[("小明", 0.85, (100, 100, 200, 200)), ...]
            位置框是 (左上角x, 左上角y, 右下角x, 右下角y)
        """
        results = []  # 创建空列表，用来存放识别结果
        
        # 【第1步】使用YOLO检测人脸位置
        # 就像用放大镜在照片上找所有的脸
        yolo_results = self.yolo_model(
            frame,
            verbose=False,
            device=self.yolo_device,
            half=self.yolo_half,
        )
        
        # 【第2步】处理每个检测到的人脸
        for result in yolo_results:
            boxes = result.boxes  # 获取所有检测到的框
            
            # 遍历每个检测到的框（每个框代表一张可能的脸）
            for box in boxes:
                # 获取这个框的信心值（YOLO有多确定这是一张脸）
                conf = float(box.conf[0])
                cls = int(box.cls[0])  # 获取类别（通常是"人脸"类别）
                
                # 如果信心值太低，跳过这个框
                # 就像："这个不太像脸，算了，不管它"
                if conf < self.confidence:
                    continue
                
                # 获取框的四个角的坐标
                # (x1, y1)是左上角，(x2, y2)是右下角
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 确保坐标不超出图片边界
                # 防止框跑到图片外面去了
                h, w = frame.shape[:2]  # 获取图片的高度和宽度
                x1, y1 = max(0, x1), max(0, y1)  # 左上角不能是负数
                x2, y2 = min(w, x2), min(h, y2)  # 右下角不能超过图片大小
                
                # 从图片中裁剪出人脸区域
                # 就像用剪刀把照片中的脸剪下来
                face_region = frame[y1:y2, x1:x2]
                
                # 检查裁剪的区域是否有效（不是空的）
                if face_region.size == 0:
                    continue
                
                # 【第3步】识别这张脸是谁
                
                # 将颜色格式从BGR转换为RGB
                # （OpenCV用BGR，face_recognition用RGB，需要转换）
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                
                # 提取这张脸的特征编码
                # 就像给这张脸生成一个独特的"指纹"
                face_encodings = face_recognition.face_encodings(rgb_face)
                
                # 如果没有提取到特征，标记为"未知"
                if len(face_encodings) == 0:
                    results.append(("未知", conf, (x1, y1, x2, y2)))
                    continue
                
                face_encoding = face_encodings[0]  # 使用第一个特征
                
                # 【第4步】和数据库中的人脸进行比对
                name = "未知"  # 默认是未知
                best_match_confidence = 0.0  # 最佳匹配的信心值
                
                # 如果数据库中有保存的人脸数据
                if len(self.known_face_encodings) > 0:
                    # 计算当前人脸和数据库中所有人脸的距离
                    # 距离越小，说明越相似
                    # 就像比较两个人长得有多像
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    
                    # 找到距离最小的（最相似的）人脸
                    best_match_idx = np.argmin(face_distances)  # 找到最小值的位置
                    best_distance = face_distances[best_match_idx]  # 最小的距离
                    
                    # 将距离转换为信心值（距离小 = 信心高）
                    # 0.6是经验阈值：距离小于0.6认为是同一个人
                    match_confidence = 1 - best_distance
                    
                    # 如果距离足够小，认为识别成功
                    if best_distance < 0.6:
                        name = self.known_face_names[best_match_idx]  # 获取这个人的名字
                        best_match_confidence = match_confidence
                
                # 把识别结果添加到结果列表
                results.append((name, best_match_confidence, (x1, y1, x2, y2)))
        
        return results  # 返回所有识别结果
    
    def draw_results(self, frame, results):
        """
        【在图片上画出识别结果】
        在每张检测到的脸周围画一个框，并标注名字
        就像给照片做标记，圈出谁是谁
        
        参数说明：
            frame: 原始图片
            results: 识别结果列表
            
        返回值：
            画好标记的图片
        """
        # 先画检测框，再统一用 Pillow 绘制中文标签，避免中文乱码
        label_tasks: List[Dict] = []

        for name, confidence, (x1, y1, x2, y2) in results:
            # 根据是否识别成功选择颜色
            if name == "未知":
                color = (0, 0, 255)  # 红色表示不认识（BGR）
            else:
                color = (0, 255, 0)  # 绿色表示认识（BGR）

            # 人脸检测框（使用 OpenCV 绘制）
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 准备标签文字（名字和信心值）
            label_text = f"{name} ({confidence:.2f})"

            label_tasks.append({
                'text': label_text,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'bg_bgr': color,
            })

        # 使用 Pillow 渲染中文标签
        frame = self._draw_labels_pil(frame, label_tasks)

        return frame  # 返回画好的图片
    
    def process_image(self, image_path, output_path=None, show=True):
        """
        【处理单张图片】
        读取一张图片，识别里面的人脸，并显示/保存结果
        
        参数说明：
            image_path: 输入图片的路径
            output_path: 保存结果图片的路径（可选）
            show: 是否显示结果窗口（True=显示，False=不显示）
            
        返回值：
            识别结果列表
        """
        # 读取图片文件
        frame = cv2.imread(image_path)
        if frame is None:
            print(f" 错误：无法读取图片 {image_path}")
            return []
        
        # 识别图片中的人脸
        results = self.recognize_faces_in_frame(frame)
        
        # 在图片副本上画出识别结果
        # 使用copy()是为了保留原图不变
        output_frame = self.draw_results(frame.copy(), results)
        
        # 如果需要显示结果
        if show:
            cv2.imshow('人脸识别结果', output_frame)  # 显示窗口
            print("按任意键关闭窗口...")
            cv2.waitKey(0)  # 等待按键
            cv2.destroyAllWindows()  # 关闭所有窗口
        
        # 如果指定了输出路径，保存结果图片
        if output_path:
            cv2.imwrite(output_path, output_frame)
            print(f" 结果已保存到 {output_path}")
        
        # 打印识别结果摘要
        print(f"\n检测到 {len(results)} 张人脸：")
        for name, conf, bbox in results:
            print(f"  - {name}（信心值：{conf:.2f}）")
        
        return results
    
    def process_camera(self, camera_id=0):
        """
        【处理摄像头实时视频】
        打开摄像头，实时识别视频中的人脸
        就像照镜子，电脑会实时告诉你看到了谁
        
        参数说明：
            camera_id: 摄像头设备编号（通常0是默认摄像头）
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print(f" 错误：无法打开摄像头 {camera_id}")
            return
        
        print(" 摄像头已打开。按 'q' 键退出。")
        
        # 处理视频帧（循环）
        frame_count = 0  # 帧计数器
        while True:
            # 读取一帧画面
            ret, frame = cap.read()
            if not ret:
                print(" 错误：无法读取摄像头画面")
                break
            
            # 每一帧都进行识别
            # 注意：如果电脑太慢，可以改成每隔几帧识别一次
            if frame_count % 1 == 0:  # 目前是每帧都识别
                results = self.recognize_faces_in_frame(frame)
                frame = self.draw_results(frame, results)
            
            frame_count += 1  # 帧计数加1
            
            # 显示画面
            cv2.imshow('人脸识别（按q退出）', frame)
            
            # 检查是否按下了 'q' 键
            # waitKey(1) 表示等待1毫秒，如果按了键就返回该键的ASCII码
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理工作
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭所有窗口
        print(" 摄像头已关闭")


def main():
    """
    【主函数】
    处理命令行输入，根据用户选择执行图片识别或摄像头识别
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLO人脸识别系统')
    parser.add_argument('--db', default='face_database.pkl', 
                       help='人脸数据库路径（默认：face_database.pkl）')
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='YOLO模型路径（默认：yolov8n.pt）')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='检测信心阈值（默认：0.5）')
    parser.add_argument('--device', default=None,
                       help='推理设备：如 0/cuda:0 或 cpu（默认自动检测）')
    parser.add_argument('--no-half', action='store_true',
                       help='禁用半精度FP16（Jetson上如遇问题可加该参数）')
    
    # 创建子命令（两种模式）
    subparsers = parser.add_subparsers(dest='mode', help='识别模式')
    
    # 【模式1：图片识别】
    image_parser = subparsers.add_parser('image', help='处理图片文件')
    image_parser.add_argument('input', help='输入图片路径')
    image_parser.add_argument('--output', help='输出图片路径')
    image_parser.add_argument('--no-show', action='store_true', 
                             help='不显示结果窗口')
    
    # 【模式2：摄像头识别】
    camera_parser = subparsers.add_parser('camera', help='处理摄像头视频')
    camera_parser.add_argument('--camera-id', type=int, default=0, 
                              help='摄像头设备编号（默认：0）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定模式，显示帮助信息
    if not args.mode:
        parser.print_help()
        return
    
    # 初始化识别器
    recognizer = YOLOFaceRecognizer(
        db_path=args.db,
        yolo_model=args.model,
        confidence=args.confidence,
        device=args.device,
        use_half=(not args.no_half)
    )
    
    # 根据模式执行相应操作
    if args.mode == 'image':
        # 图片识别模式
        recognizer.process_image(
            args.input,
            output_path=args.output,
            show=not args.no_show
        )
    
    elif args.mode == 'camera':
        # 摄像头识别模式
        recognizer.process_camera(camera_id=args.camera_id)


# 程序启动代码
if __name__ == '__main__':
    main()