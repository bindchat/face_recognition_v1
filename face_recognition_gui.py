#!/usr/bin/env python3
"""
【人脸识别图形界面程序】
这是一个带图形界面的人脸识别系统
不需要敲命令，只需要点击按钮就能使用
就像玩游戏一样简单！
"""

# 导入需要的工具包
import tkinter as tk  # Tkinter - Python自带的图形界面工具
from tkinter import ttk, filedialog, messagebox, scrolledtext  # 各种界面组件
import threading  # 多线程工具，让程序不会卡住
import cv2  # OpenCV - 图像处理工具
from PIL import Image, ImageTk  # 图像处理工具，用来在界面上显示图片
import os  # 文件操作工具
from face_database import FaceDatabase  # 人脸数据库管理
from face_recognition_yolo import YOLOFaceRecognizer  # 人脸识别器


class FaceRecognitionGUI:
    """
    【人脸识别图形界面类】
    这是整个图形界面的核心，包含所有功能
    就像一个操作面板，有各种按钮和显示区域
    """
    
    def __init__(self, root):
        """
        【初始化界面】
        创建整个界面的布局和所有按钮
        
        参数说明：
            root: 主窗口对象（Tkinter的根窗口）
        """
        self.root = root
        self.root.title("🎯 人脸识别系统 - 图形界面")  # 设置窗口标题
        self.root.geometry("900x700")  # 设置窗口大小（宽x高）
        
        # 设置变量
        self.db_path = "face_database.pkl"  # 数据库文件路径
        self.recognizer = None  # 识别器对象（暂时为空）
        self.camera_running = False  # 摄像头是否正在运行
        self.camera_thread = None  # 摄像头线程
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """
        【创建界面组件】
        这个函数创建所有的按钮、文本框、标签等界面元素
        就像搭积木一样，把各个部分组装起来
        """
        
        # ============ 顶部标题区域 ============
        title_frame = tk.Frame(self.root, bg="#4A90E2", height=80)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        
        # 主标题
        title_label = tk.Label(
            title_frame, 
            text="🎯 人脸识别系统", 
            font=("Arial", 24, "bold"),
            bg="#4A90E2", 
            fg="white"
        )
        title_label.pack(pady=20)
        
        # ============ 主内容区域 ============
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        left_frame = tk.Frame(main_frame, width=300, relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        
        # 右侧显示区域
        right_frame = tk.Frame(main_frame, relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # ============ 左侧面板内容 ============
        
        # --- 数据库管理区域 ---
        db_label = tk.Label(
            left_frame, 
            text="📁 数据库管理", 
            font=("Arial", 14, "bold"),
            bg="#E8F4F8",
            pady=5
        )
        db_label.pack(fill=tk.X, pady=(10, 5))
        
        # 按钮1：从文件夹导入人脸
        self.import_btn = tk.Button(
            left_frame,
            text="📂 从文件夹导入人脸",
            command=self.import_faces,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2
        )
        self.import_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # 按钮2：添加单张人脸
        self.add_face_btn = tk.Button(
            left_frame,
            text="➕ 添加单张人脸",
            command=self.add_single_face,
            bg="#2196F3",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2
        )
        self.add_face_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # 按钮3：查看数据库
        self.view_db_btn = tk.Button(
            left_frame,
            text="👁️ 查看数据库",
            command=self.view_database,
            bg="#FF9800",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2
        )
        self.view_db_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # 按钮4：清空数据库
        self.clear_db_btn = tk.Button(
            left_frame,
            text="🗑️ 清空数据库",
            command=self.clear_database,
            bg="#F44336",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2
        )
        self.clear_db_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # 分隔线
        separator1 = ttk.Separator(left_frame, orient=tk.HORIZONTAL)
        separator1.pack(fill=tk.X, padx=10, pady=15)
        
        # --- 人脸识别区域 ---
        recog_label = tk.Label(
            left_frame,
            text="🔍 人脸识别",
            font=("Arial", 14, "bold"),
            bg="#FFF3E0",
            pady=5
        )
        recog_label.pack(fill=tk.X, pady=(5, 5))
        
        # 按钮5：识别图片
        self.recog_image_btn = tk.Button(
            left_frame,
            text="🖼️ 识别图片中的人脸",
            command=self.recognize_image,
            bg="#9C27B0",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2
        )
        self.recog_image_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # 按钮6：打开摄像头
        self.camera_btn = tk.Button(
            left_frame,
            text="📷 打开摄像头识别",
            command=self.toggle_camera,
            bg="#00BCD4",
            fg="white",
            font=("Arial", 11),
            cursor="hand2",
            relief=tk.RAISED,
            borderwidth=2
        )
        self.camera_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # 分隔线
        separator2 = ttk.Separator(left_frame, orient=tk.HORIZONTAL)
        separator2.pack(fill=tk.X, padx=10, pady=15)
        
        # --- 设置区域 ---
        settings_label = tk.Label(
            left_frame,
            text="⚙️ 设置",
            font=("Arial", 14, "bold"),
            bg="#E8EAF6",
            pady=5
        )
        settings_label.pack(fill=tk.X, pady=(5, 5))
        
        # 信心值设置
        confidence_frame = tk.Frame(left_frame)
        confidence_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            confidence_frame,
            text="检测信心值:",
            font=("Arial", 10)
        ).pack(side=tk.LEFT)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = tk.Scale(
            confidence_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var
        )
        self.confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ============ 右侧显示区域 ============
        
        # 显示区域标题
        display_title = tk.Label(
            right_frame,
            text="📺 显示区域",
            font=("Arial", 14, "bold"),
            bg="#F5F5F5",
            pady=5
        )
        display_title.pack(fill=tk.X)
        
        # 图片显示画布
        self.canvas = tk.Canvas(right_frame, bg="gray", height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 提示文字（初始状态）
        self.canvas.create_text(
            300, 200,
            text="👆 点击左侧按钮开始使用",
            font=("Arial", 16),
            fill="white",
            tags="hint"
        )
        
        # 日志输出区域
        log_label = tk.Label(
            right_frame,
            text="📋 操作日志",
            font=("Arial", 12, "bold"),
            bg="#FFFDE7",
            pady=5
        )
        log_label.pack(fill=tk.X)
        
        # 滚动文本框（显示操作日志）
        self.log_text = scrolledtext.ScrolledText(
            right_frame,
            height=10,
            font=("Courier", 9),
            bg="#FAFAFA"
        )
        self.log_text.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # 显示欢迎信息
        self.log("=" * 50)
        self.log("✨ 欢迎使用人脸识别系统！")
        self.log("💡 提示：请先导入人脸数据库，然后就可以开始识别了")
        self.log("=" * 50)
    
    def log(self, message):
        """
        【记录日志】
        在界面上显示操作信息
        就像写日记一样，记录程序做了什么
        
        参数说明：
            message: 要显示的消息
        """
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # 自动滚动到最新的日志
        self.root.update()  # 更新界面显示
    
    def import_faces(self):
        """
        【从文件夹导入人脸】
        让用户选择一个文件夹，然后批量导入人脸照片
        
        文件夹结构应该是：
        选择的文件夹/
            小明/
                照片1.jpg
            小红/
                照片1.jpg
        """
        self.log("\n📂 准备导入人脸...")
        
        # 打开文件夹选择对话框
        directory = filedialog.askdirectory(title="选择包含人脸照片的文件夹")
        
        # 如果用户没有选择文件夹（点了取消）
        if not directory:
            self.log("❌ 已取消导入")
            return
        
        self.log(f"📁 选择的文件夹: {directory}")
        self.log("⏳ 正在导入，请稍候...")
        
        try:
            # 创建数据库对象并导入
            db = FaceDatabase(self.db_path)
            db.import_from_directory(directory)
            
            self.log("✅ 导入完成！")
            messagebox.showinfo("成功", "人脸导入成功！")
        except Exception as e:
            self.log(f"❌ 导入失败: {str(e)}")
            messagebox.showerror("错误", f"导入失败：{str(e)}")
    
    def add_single_face(self):
        """
        【添加单张人脸】
        让用户选择一张照片，并输入这个人的名字
        然后把这张人脸添加到数据库
        """
        self.log("\n➕ 准备添加单张人脸...")
        
        # 打开文件选择对话框
        image_path = filedialog.askopenfilename(
            title="选择人脸照片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        # 如果用户没有选择文件
        if not image_path:
            self.log("❌ 已取消添加")
            return
        
        # 弹出对话框，让用户输入名字
        name = tk.simpledialog.askstring("输入名字", "请输入这个人的名字：")
        
        if not name:
            self.log("❌ 已取消添加（未输入名字）")
            return
        
        self.log(f"📄 图片: {os.path.basename(image_path)}")
        self.log(f"👤 名字: {name}")
        self.log("⏳ 正在处理...")
        
        try:
            # 创建数据库对象并添加人脸
            db = FaceDatabase(self.db_path)
            success = db.add_face_from_image(image_path, name)
            
            if success:
                db.save_database()
                self.log("✅ 添加成功！")
                messagebox.showinfo("成功", f"已成功添加 {name} 的人脸！")
            else:
                self.log("❌ 添加失败（可能图片中没有检测到人脸）")
                messagebox.showwarning("失败", "添加失败，请确保图片中有清晰的人脸")
        except Exception as e:
            self.log(f"❌ 添加失败: {str(e)}")
            messagebox.showerror("错误", f"添加失败：{str(e)}")
    
    def view_database(self):
        """
        【查看数据库】
        显示数据库中保存了哪些人的人脸
        就像查看通讯录一样
        """
        self.log("\n👁️ 查看数据库...")
        
        try:
            # 加载数据库
            db = FaceDatabase(self.db_path)
            
            if not db.face_names:
                self.log("📭 数据库是空的")
                messagebox.showinfo("数据库信息", "数据库中还没有任何人脸数据")
                return
            
            # 统计每个人的照片数量
            from collections import Counter
            name_counts = Counter(db.face_names)
            
            # 生成显示信息
            info = f"数据库中共有 {len(db.face_names)} 张人脸照片\n\n"
            info += "详细信息：\n"
            info += "-" * 30 + "\n"
            for name, count in sorted(name_counts.items()):
                info += f"👤 {name}: {count} 张照片\n"
                self.log(f"👤 {name}: {count} 张照片")
            
            # 显示信息对话框
            messagebox.showinfo("数据库信息", info)
            
        except Exception as e:
            self.log(f"❌ 查看失败: {str(e)}")
            messagebox.showerror("错误", f"查看失败：{str(e)}")
    
    def clear_database(self):
        """
        【清空数据库】
        删除数据库中的所有人脸数据
        注意：这个操作很危险，不能撤销！
        """
        self.log("\n🗑️ 准备清空数据库...")
        
        # 弹出确认对话框
        result = messagebox.askyesno(
            "⚠️ 危险操作",
            "确定要清空数据库吗？\n\n这将删除所有人脸数据，且无法恢复！",
            icon="warning"
        )
        
        if not result:
            self.log("❌ 已取消清空操作")
            return
        
        try:
            # 清空数据库
            db = FaceDatabase(self.db_path)
            db.clear_database()
            
            self.log("✅ 数据库已清空")
            messagebox.showinfo("完成", "数据库已清空")
        except Exception as e:
            self.log(f"❌ 清空失败: {str(e)}")
            messagebox.showerror("错误", f"清空失败：{str(e)}")
    
    def recognize_image(self):
        """
        【识别图片中的人脸】
        让用户选择一张照片，程序会识别照片中的人是谁
        并在照片上画框和标注名字
        """
        self.log("\n🖼️ 准备识别图片...")
        
        # 打开文件选择对话框
        image_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        if not image_path:
            self.log("❌ 已取消识别")
            return
        
        self.log(f"📄 图片: {os.path.basename(image_path)}")
        self.log("⏳ 正在识别...")
        
        try:
            # 初始化识别器（如果还没有初始化）
            if self.recognizer is None:
                self.log("🔧 正在初始化识别器...")
                self.recognizer = YOLOFaceRecognizer(
                    db_path=self.db_path,
                    confidence=self.confidence_var.get()
                )
            
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                self.log("❌ 无法读取图片")
                messagebox.showerror("错误", "无法读取图片文件")
                return
            
            # 识别人脸
            results = self.recognizer.recognize_faces_in_frame(frame)
            
            # 在图片上画框和标注
            output_frame = self.recognizer.draw_results(frame, results)
            
            # 在界面上显示结果图片
            self.display_image(output_frame)
            
            # 显示识别结果
            self.log(f"✅ 识别完成！检测到 {len(results)} 张人脸：")
            for name, conf, bbox in results:
                self.log(f"   👤 {name} (信心值: {conf:.2f})")
            
            # 弹出结果信息
            if len(results) == 0:
                messagebox.showinfo("识别结果", "图片中没有检测到人脸")
            else:
                result_text = f"检测到 {len(results)} 张人脸：\n\n"
                for name, conf, bbox in results:
                    result_text += f"👤 {name} (信心值: {conf:.2f})\n"
                messagebox.showinfo("识别结果", result_text)
                
        except Exception as e:
            self.log(f"❌ 识别失败: {str(e)}")
            messagebox.showerror("错误", f"识别失败：{str(e)}")
    
    def toggle_camera(self):
        """
        【开关摄像头】
        如果摄像头没开，就打开它
        如果摄像头已经开了，就关闭它
        """
        if not self.camera_running:
            # 打开摄像头
            self.start_camera()
        else:
            # 关闭摄像头
            self.stop_camera()
    
    def start_camera(self):
        """
        【启动摄像头识别】
        打开摄像头，实时识别视频中的人脸
        """
        self.log("\n📷 正在启动摄像头...")
        
        try:
            # 初始化识别器（如果还没有初始化）
            if self.recognizer is None:
                self.log("🔧 正在初始化识别器...")
                self.recognizer = YOLOFaceRecognizer(
                    db_path=self.db_path,
                    confidence=self.confidence_var.get()
                )
            
            # 设置状态
            self.camera_running = True
            self.camera_btn.config(text="⏹️ 关闭摄像头", bg="#F44336")
            
            # 在新线程中运行摄像头（避免界面卡住）
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.log("✅ 摄像头已启动")
            
        except Exception as e:
            self.log(f"❌ 启动失败: {str(e)}")
            messagebox.showerror("错误", f"启动摄像头失败：{str(e)}")
            self.camera_running = False
    
    def stop_camera(self):
        """
        【停止摄像头】
        关闭摄像头，停止识别
        """
        self.log("\n⏹️ 正在关闭摄像头...")
        self.camera_running = False
        self.camera_btn.config(text="📷 打开摄像头识别", bg="#00BCD4")
        self.log("✅ 摄像头已关闭")
    
    def camera_loop(self):
        """
        【摄像头循环】
        持续读取摄像头画面并识别人脸
        这个函数在单独的线程中运行，不会卡住界面
        """
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.log("❌ 无法打开摄像头")
            self.camera_running = False
            return
        
        frame_count = 0
        
        # 持续读取画面
        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每一帧都识别（如果电脑慢可以改成每隔几帧识别一次）
            if frame_count % 1 == 0:
                try:
                    results = self.recognizer.recognize_faces_in_frame(frame)
                    frame = self.recognizer.draw_results(frame, results)
                except:
                    pass
            
            frame_count += 1
            
            # 在界面上显示
            self.display_image(frame)
        
        # 释放摄像头
        cap.release()
    
    def display_image(self, cv_image):
        """
        【在界面上显示图片】
        把OpenCV的图片转换成Tkinter能显示的格式
        然后显示在界面的画布上
        
        参数说明：
            cv_image: OpenCV格式的图片（BGR）
        """
        try:
            # 转换颜色格式（BGR -> RGB）
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 调整图片大小以适应画布
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 640, 480
            
            # 计算缩放比例（保持宽高比）
            h, w = rgb_image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.95
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 调整大小
            resized = cv2.resize(rgb_image, (new_w, new_h))
            
            # 转换为PIL图片
            pil_image = Image.fromarray(resized)
            
            # 转换为Tkinter图片
            self.photo = ImageTk.PhotoImage(image=pil_image)
            
            # 清空画布并显示图片
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo,
                anchor=tk.CENTER
            )
            
        except Exception as e:
            pass  # 忽略显示错误
    
    def on_closing(self):
        """
        【关闭窗口时的处理】
        当用户关闭窗口时，需要先停止摄像头
        """
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()


def main():
    """
    【主函数】
    创建图形界面并启动程序
    """
    # 创建主窗口
    root = tk.Tk()
    
    # 创建GUI对象
    app = FaceRecognitionGUI(root)
    
    # 设置关闭窗口的处理函数
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动主循环（程序会一直运行，直到关闭窗口）
    root.mainloop()


# 程序启动代码
if __name__ == '__main__':
    # 需要导入简单对话框工具
    import tkinter.simpledialog
    main()