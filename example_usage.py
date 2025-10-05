#!/usr/bin/env python3
"""
【示例程序 - 教你如何使用人脸识别系统】
这个文件就像一本使用说明书，教你怎么用这个人脸识别程序
"""

# 导入（引入）需要用到的工具
# 就像打开工具箱，拿出需要的工具一样
from face_database import FaceDatabase  # 人脸数据库工具
from face_recognition_yolo import YOLOFaceRecognizer  # 人脸识别工具


def example_database_management():
    """
    【示例1：管理人脸数据库】
    这个函数教你怎么管理保存人脸照片的数据库
    就像管理一个相册，可以添加照片、查看照片等
    """
    print("=" * 50)  # 打印50个等号，让界面更好看
    print("示例1：数据库管理")
    print("=" * 50)
    
    # 创建一个数据库对象
    # 就像创建一个新的相册本子，取名叫'example_database.pkl'
    db = FaceDatabase('example_database.pkl')
    
    # 下面是一些可以做的操作（被注释掉了，需要的时候可以取消注释）：
    
    # 添加一张人脸照片到数据库
    # 就像往相册里贴一张照片，并写上这个人的名字
    # db.add_face_from_image('照片路径/照片.jpg', '人的名字')
    
    # 从一个文件夹批量导入人脸照片
    # 就像把一整本相册的照片都复制到新相册里
    # db.import_from_directory('faces/')
    
    # 列出数据库里所有的人脸
    # 就像翻看相册目录，看看里面都有谁的照片
    db.list_faces()
    
    print("\n")  # 打印一个空行，让界面整洁


def example_image_recognition():
    """
    【示例2：识别图片中的人脸】
    这个函数教你怎么识别照片里的人是谁
    就像看着一张照片说："哦，这是小明！"
    """
    print("=" * 50)
    print("示例2：图片识别")
    print("=" * 50)
    
    # 初始化（准备好）识别器
    # 就像准备好放大镜和相册，开始找人的过程
    recognizer = YOLOFaceRecognizer(
        db_path='face_database.pkl',  # 人脸数据库的位置
        yolo_model='yolov8n.pt',      # 用来找脸的AI模型（就像一个聪明的机器人）
        confidence=0.5                # 信心值：0.5表示有50%以上把握才算找到脸
    )
    
    # 处理一张图片（被注释掉的示例代码）
    # results = recognizer.process_image(
    #     'test_image.jpg',      # 要识别的图片
    #     output_path='result.jpg',  # 结果保存的位置
    #     show=True              # 是否显示结果
    # )
    
    # 结果的格式：每个人的 [名字, 信心值, 脸的位置(框)]
    # for name, conf, (x1, y1, x2, y2) in results:
    #     print(f"找到了：{name}，信心值 {conf:.2f}")
    
    print("初始化一个识别器，然后调用 process_image() 来识别图片")
    print("\n")


def example_camera_recognition():
    """
    【示例3：实时摄像头识别】
    这个函数教你怎么用摄像头实时识别人脸
    就像在镜子里看自己，电脑会告诉你"这是XXX"
    """
    print("=" * 50)
    print("示例3：摄像头识别")
    print("=" * 50)
    
    # 初始化识别器（准备工作，和示例2一样）
    # recognizer = YOLOFaceRecognizer(
    #     db_path='face_database.pkl',
    #     yolo_model='yolov8n.pt',
    #     confidence=0.5
    # )
    
    # 开始摄像头识别
    # camera_id=0 表示使用第1个摄像头（电脑通常只有1个摄像头）
    # recognizer.process_camera(camera_id=0)
    
    print("初始化识别器后调用 process_camera() 即可开始")
    print("按'q'键可以退出摄像头画面")
    print("\n")


def main():
    """
    【主函数】
    这是程序的入口，就像故事的开头
    当你运行这个程序时，就会从这里开始执行
    """
    print("\n")
    # 打印一个漂亮的标题框
    print("╔" + "=" * 60 + "╗")
    print("║" + " " * 10 + "YOLO 人脸识别系统 - 使用示例" + " " * 23 + "║")
    print("╚" + "=" * 60 + "╝")
    print("\n")
    
    # 依次运行三个示例
    example_database_management()    # 示例1：数据库管理
    example_image_recognition()      # 示例2：图片识别
    example_camera_recognition()     # 示例3：摄像头识别
    
    # 打印快速入门指南
    print("=" * 50)
    print("【快速入门指南】")
    print("=" * 50)
    print("""
1. 先准备好人脸图片，按照下面的文件夹结构组织：
   （就像整理相册，每个人的照片放在一个文件夹里）
   faces/
   ├── 小明/
   │   ├── 照片1.jpg
   │   └── 照片2.jpg
   └── 小红/
       └── 照片1.jpg

2. 把人脸导入到数据库：
   （把相册里的照片信息记录到电脑里）
   $ python face_database.py import faces/

3. 识别图片中的人脸：
   （让电脑看一张照片，告诉你里面是谁）
   $ python recognize_image.py 照片.jpg

4. 用摄像头实时识别：
   （打开摄像头，电脑会实时告诉你看到了谁）
   $ python recognize_camera.py

想了解更多信息，请查看 README.md 文件
    """)


# 这是Python程序的启动代码
# 如果直接运行这个文件，就会执行main()函数
if __name__ == '__main__':
    main()