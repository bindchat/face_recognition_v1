#!/usr/bin/env python3
"""
【图片人脸识别程序】
这是一个简单易用的图片识别工具
可以识别照片中的人脸，告诉你照片里是谁
就像一个智能的相册管理器，能自动给照片里的人打标签
"""

# 导入需要的工具
import sys  # 系统工具
import argparse  # 命令行参数解析器
import os  # 读取环境变量以获取中文字体路径
from face_recognition_yolo import YOLOFaceRecognizer  # 导入人脸识别器


def main():
    """
    【主函数】
    这是程序的入口，负责：
    1. 接收用户输入的参数（比如要识别哪张照片）
    2. 启动人脸识别系统
    3. 识别照片中的人脸并显示结果
    """
    
    # 创建命令行参数解析器
    # 这个工具可以理解用户在命令行输入的各种选项
    parser = argparse.ArgumentParser(
        description='使用YOLO识别图片中的人脸',  # 程序说明
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 保留格式
        epilog="""
【使用示例】
  基本用法（识别一张照片）：
  python recognize_image.py 照片.jpg
  
  识别照片并保存结果：
  python recognize_image.py 照片.jpg --output 结果.jpg
  
  使用自己的数据库和调整信心值：
  python recognize_image.py 照片.jpg --db my_faces.pkl --confidence 0.6
        """
    )
    
    # 添加各种参数（就像给程序提供不同的选项）
    
    # 必需参数：要识别的图片文件
    # 这是唯一必须提供的参数，其他都是可选的
    parser.add_argument('image', help='要识别的图片文件路径（必需）')
    
    # 可选参数1：结果图片保存路径
    parser.add_argument('--output', '-o', 
                       help='保存识别结果的图片路径（可选）')
    
    # 可选参数2：人脸数据库文件的位置
    parser.add_argument('--db', default='face_database.pkl', 
                       help='人脸数据库文件路径（默认：face_database.pkl）')
    
    # 可选参数3：YOLO模型文件的位置
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='YOLO模型文件路径（默认：yolov8n.pt）')
    
    # 可选参数4：检测的信心阈值
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='检测信心阈值，范围0-1（默认：0.5表示50%把握）')
    
    # 可选参数5：是否显示结果窗口
    parser.add_argument('--no-show', action='store_true', 
                       help='不显示结果窗口（只保存不显示）')

    # 可选参数6：GPU/精度
    parser.add_argument('--device', default=None,
                       help='推理设备：如 0/cuda:0 或 cpu（默认自动）')
    parser.add_argument('--no-half', action='store_true',
                       help='禁用半精度FP16（Jetson如遇不兼容时使用）')

    # 可选参数6：中文字体路径与字号
    parser.add_argument('--font', help='中文字体文件路径（如 NotoSansCJK 或思源黑体）')
    parser.add_argument('--font-size', type=int, default=20, help='标签文字字号（默认：20）')
    
    # 解析用户输入的参数
    args = parser.parse_args()
    
    # 【第1步】初始化人脸识别系统
    print("正在初始化人脸识别系统...")
    print("（第一次运行时可能需要下载AI模型，请耐心等待）")
    
    # 创建识别器对象，传入用户指定的参数
    # 支持中文字体路径通过环境变量或默认路径自动发现
    recognizer = YOLOFaceRecognizer(
        db_path=args.db,              # 使用哪个人脸数据库
        yolo_model=args.model,        # 使用哪个YOLO模型
        confidence=args.confidence,   # 检测的信心要求
        font_path=(args.font or os.getenv('CHINESE_FONT_PATH')),
        font_size=args.font_size,
        device=args.device,
        use_half=(not args.no_half)
    )
    
    # 【第2步】处理图片
    print(f"\n正在处理图片：{args.image}")
    print("程序会：")
    print("  1. 在图片中找到所有人脸")
    print("  2. 识别每张脸是谁")
    print("  3. 在脸周围画框并标注名字\n")
    
    # 调用识别器的图片处理函数
    recognizer.process_image(
        args.image,                   # 输入图片路径
        output_path=args.output,      # 输出图片路径（如果指定了的话）
        show=not args.no_show         # 是否显示结果（如果没有--no-show就显示）
    )


# 程序启动代码
# 这是Python的标准写法，表示"如果直接运行这个文件，就执行main()函数"
# 如果这个文件被其他程序导入（import），则不会自动执行main()
if __name__ == '__main__':
    main()