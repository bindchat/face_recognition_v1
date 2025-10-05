#!/usr/bin/env python3
"""
【摄像头人脸识别程序】
这是一个简单易用的摄像头识别工具
可以快速启动摄像头，实时识别视频中的人脸
就像一个智能的魔镜，能认出照镜子的人是谁
"""

# 导入需要的工具
import sys  # 系统工具
import argparse  # 命令行参数解析器
from face_recognition_yolo import YOLOFaceRecognizer  # 导入人脸识别器


def main():
    """
    【主函数】
    这是程序的入口，负责：
    1. 接收用户输入的参数（比如用哪个摄像头）
    2. 启动人脸识别系统
    3. 开始实时识别
    """
    
    # 创建命令行参数解析器
    # 这个工具可以理解用户在命令行输入的各种选项
    parser = argparse.ArgumentParser(
        description='使用YOLO进行实时摄像头人脸识别',  # 程序说明
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 保留格式
        epilog="""
【使用示例】
  基本用法（使用默认设置）：
  python recognize_camera.py
  
  使用第2个摄像头（如果你有多个摄像头）：
  python recognize_camera.py --camera-id 1
  
  使用自己的数据库和调整信心值：
  python recognize_camera.py --db my_faces.pkl --confidence 0.6
  
【控制方法】
  按 'q' 键 = 退出程序
        """
    )
    
    # 添加各种可选参数（就像给程序提供不同的选项）
    
    # 参数1：选择使用哪个摄像头
    parser.add_argument('--camera-id', type=int, default=0, 
                       help='摄像头设备编号（默认：0表示第一个摄像头）')
    
    # 参数2：人脸数据库文件的位置
    parser.add_argument('--db', default='face_database.pkl', 
                       help='人脸数据库文件路径（默认：face_database.pkl）')
    
    # 参数3：YOLO模型文件的位置
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='YOLO模型文件路径（默认：yolov8n.pt）')
    
    # 参数4：检测的信心阈值
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='检测信心阈值，范围0-1（默认：0.5表示50%把握）')
    
    # 解析用户输入的参数
    args = parser.parse_args()
    
    # 【第1步】初始化人脸识别系统
    print("正在初始化人脸识别系统...")
    print("（第一次运行时可能需要下载AI模型，请耐心等待）")
    
    # 创建识别器对象，传入用户指定的参数
    recognizer = YOLOFaceRecognizer(
        db_path=args.db,              # 使用哪个人脸数据库
        yolo_model=args.model,        # 使用哪个YOLO模型
        confidence=args.confidence    # 检测的信心要求
    )
    
    # 【第2步】开始摄像头识别
    print(f"\n正在启动摄像头 {args.camera_id}...")
    print("提示：摄像头启动后，把脸对准摄像头，程序会自动识别你是谁")
    print("      按 'q' 键可以随时退出程序\n")
    
    # 调用识别器的摄像头处理函数
    # 这个函数会一直运行，直到你按'q'键退出
    recognizer.process_camera(camera_id=args.camera_id)


# 程序启动代码
# 这是Python的标准写法，表示"如果直接运行这个文件，就执行main()函数"
# 如果这个文件被其他程序导入（import），则不会自动执行main()
if __name__ == '__main__':
    main()