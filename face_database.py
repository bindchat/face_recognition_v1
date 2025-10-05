#!/usr/bin/env python3
"""
【人脸数据库管理工具】
这个程序用来管理人脸数据库，就像管理一个电子相册
可以把人脸照片导入进来，保存每个人的面部特征信息
"""

# 导入需要的工具包（就像打开工具箱）
import os  # 操作系统工具，用来处理文件和文件夹
import pickle  # 腌制工具，用来保存和读取数据（pickle在英语里是腌菜的意思）
import argparse  # 参数解析器，用来处理命令行输入
import face_recognition  # 人脸识别工具，用来识别和分析人脸
import cv2  # OpenCV图像处理库，用来读取和处理图片
from pathlib import Path  # 路径工具，用来处理文件路径


class FaceDatabase:
    """
    【人脸数据库类】
    这是一个管理人脸数据的"类"，就像一个电子相册管理器
    可以添加人脸、保存人脸、查看人脸等功能
    """
    
    def __init__(self, db_path='face_database.pkl'):
        """
        【初始化函数 - 创建数据库对象】
        当创建一个新的人脸数据库时，这个函数会被自动调用
        
        参数说明：
            db_path: 数据库文件保存的位置（默认是'face_database.pkl'）
        """
        self.db_path = db_path  # 保存数据库文件的路径
        self.face_encodings = []  # 存放所有人脸特征的列表（空列表，等待添加）
        self.face_names = []  # 存放所有人脸名字的列表
        self.load_database()  # 尝试加载已有的数据库
    
    def load_database(self):
        """
        【加载数据库】
        从文件中读取已经保存的人脸数据
        就像打开一本相册，看看里面已经有什么照片了
        """
        # 检查数据库文件是否存在
        if os.path.exists(self.db_path):
            try:
                # 打开文件并读取数据
                with open(self.db_path, 'rb') as f:  # 'rb'表示以二进制读取模式打开
                    data = pickle.load(f)  # 从文件中读取数据
                    # 获取人脸特征编码（就像每个人独特的"身份证"）
                    self.face_encodings = data.get('encodings', [])
                    # 获取人脸对应的名字
                    self.face_names = data.get('names', [])
                print(f"✓ 成功加载了 {len(self.face_names)} 张人脸数据")
            except Exception as e:
                # 如果出错了，打印错误信息
                print(f"加载数据库时出错了：{e}")
                # 重新创建空列表
                self.face_encodings = []
                self.face_names = []
        else:
            # 如果文件不存在，说明是第一次使用
            print("没有找到现有的数据库，正在创建新数据库。")
    
    def save_database(self):
        """
        【保存数据库】
        把人脸数据保存到文件中
        就像把相册整理好，放到书架上保存
        """
        # 把数据整理成一个字典（就像一个有标签的盒子）
        data = {
            'encodings': self.face_encodings,  # 人脸特征数据
            'names': self.face_names  # 人脸名字
        }
        try:
            # 打开文件并写入数据
            with open(self.db_path, 'wb') as f:  # 'wb'表示以二进制写入模式打开
                pickle.dump(data, f)  # 把数据保存到文件
            print(f"✓ 数据库保存成功！共有 {len(self.face_names)} 张人脸")
            return True
        except Exception as e:
            print(f"保存数据库时出错了：{e}")
            return False
    
    def add_face_from_image(self, image_path, name):
        """
        【从图片添加人脸】
        从一张照片中提取人脸特征，并保存到数据库
        就像拍一张照片，然后把这个人的特征记录下来
        
        参数说明：
            image_path: 图片文件的路径
            name: 这个人的名字
        
        返回值：
            True 表示成功，False 表示失败
        """
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"✗ 找不到图片：{image_path}")
            return False
        
        try:
            # 加载图片文件
            image = face_recognition.load_image_file(image_path)
            
            # 在图片中寻找人脸的位置
            # 就像在一张集体照中圈出每个人的脸
            face_locations = face_recognition.face_locations(image)
            
            # 提取每个人脸的特征编码
            # 就像给每张脸生成一个独特的"指纹"
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # 检查是否找到了人脸
            if len(face_encodings) == 0:
                print(f"✗ 在图片中没有检测到人脸：{image_path}")
                return False
            
            # 如果图片中有多张脸，给出提示
            if len(face_encodings) > 1:
                print(f"⚠ 图片中检测到多张人脸：{image_path}，将使用第一张脸")
            
            # 把第一张脸的特征添加到数据库
            self.face_encodings.append(face_encodings[0])
            self.face_names.append(name)
            print(f"✓ 成功添加了 '{name}' 的人脸，来自 {image_path}")
            return True
            
        except Exception as e:
            print(f"✗ 处理图片时出错：{image_path}：{e}")
            return False
    
    def import_from_directory(self, directory_path):
        """
        【从文件夹批量导入人脸】
        从一个特殊结构的文件夹中批量导入人脸照片
        
        文件夹结构示例：
        faces/              （主文件夹）
            小明/           （每个人一个文件夹）
                照片1.jpg    （这个人的照片）
                照片2.jpg
            小红/
                照片1.jpg
        
        参数说明：
            directory_path: 包含人脸照片的文件夹路径
        """
        # 检查文件夹是否存在
        if not os.path.exists(directory_path):
            print(f"✗ 找不到文件夹：{directory_path}")
            return
        
        directory = Path(directory_path)  # 创建路径对象
        added_count = 0  # 记录成功添加的人脸数量
        
        # 遍历文件夹中的每个子文件夹
        # 每个子文件夹代表一个人
        for person_dir in directory.iterdir():
            if person_dir.is_dir():  # 确保是文件夹
                person_name = person_dir.name  # 文件夹名字就是人的名字
                print(f"\n正在处理 {person_name} 的照片...")
                
                # 处理这个人文件夹中的每张图片
                for image_file in person_dir.iterdir():
                    # 检查是否是图片文件（根据后缀名判断）
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        # 尝试添加这张照片中的人脸
                        if self.add_face_from_image(str(image_file), person_name):
                            added_count += 1  # 成功添加，计数器+1
        
        print(f"\n✓ 导入完成！成功添加了 {added_count} 张人脸")
        self.save_database()  # 保存到数据库文件
    
    def list_faces(self):
        """
        【列出所有人脸】
        显示数据库中所有保存的人脸信息
        就像翻开相册目录，看看里面都有谁
        """
        # 检查数据库是否为空
        if not self.face_names:
            print("数据库是空的，还没有添加任何人脸")
            return
        
        print(f"\n数据库中的人脸（共 {len(self.face_names)} 个）：")
        
        # 统计每个人有多少张照片
        from collections import Counter  # 导入计数器工具
        name_counts = Counter(self.face_names)  # 统计每个名字出现的次数
        
        # 按字母顺序显示每个人及其照片数量
        for name, count in sorted(name_counts.items()):
            print(f"  - {name}：{count} 张照片")
    
    def clear_database(self):
        """
        【清空数据库】
        删除数据库中的所有人脸数据
        就像把相册里的照片全部清空
        注意：这个操作不能撤销！
        """
        self.face_encodings = []  # 清空人脸特征列表
        self.face_names = []  # 清空人脸名字列表
        self.save_database()  # 保存空数据库
        print("✓ 数据库已清空")


def main():
    """
    【主函数】
    这是程序的入口，处理命令行输入并执行相应的操作
    """
    # 创建命令行参数解析器
    # 就像一个接待员，理解你想做什么
    parser = argparse.ArgumentParser(description='人脸数据库管理工具')
    parser.add_argument('--db', default='face_database.pkl', 
                       help='数据库文件路径（默认：face_database.pkl）')
    
    # 创建子命令（不同的操作）
    subparsers = parser.add_subparsers(dest='command', help='可用的命令')
    
    # 【命令1：add - 添加单张人脸】
    add_parser = subparsers.add_parser('add', help='从图片添加一张人脸')
    add_parser.add_argument('image', help='图片文件路径')
    add_parser.add_argument('name', help='人的名字')
    
    # 【命令2：import - 批量导入】
    import_parser = subparsers.add_parser('import', help='从文件夹批量导入人脸')
    import_parser.add_argument('directory', help='包含人脸照片的文件夹路径')
    
    # 【命令3：list - 列出所有人脸】
    list_parser = subparsers.add_parser('list', help='列出数据库中的所有人脸')
    
    # 【命令4：clear - 清空数据库】
    clear_parser = subparsers.add_parser('clear', help='清空数据库')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有输入命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return
    
    # 创建数据库对象
    db = FaceDatabase(args.db)
    
    # 根据不同的命令执行相应的操作
    if args.command == 'add':
        # 添加单张人脸
        db.add_face_from_image(args.image, args.name)
        db.save_database()
    
    elif args.command == 'import':
        # 批量导入人脸
        db.import_from_directory(args.directory)
    
    elif args.command == 'list':
        # 列出所有人脸
        db.list_faces()
    
    elif args.command == 'clear':
        # 清空数据库（需要确认）
        response = input("确定要清空数据库吗？这个操作不能撤销！(输入yes确认): ")
        if response.lower() == 'yes':
            db.clear_database()


# 程序启动代码
# 如果直接运行这个文件，就会执行main()函数
if __name__ == '__main__':
    main()