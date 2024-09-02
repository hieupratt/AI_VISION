from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication, QStackedWidget, QFileDialog, QLabel
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
import sys
import mediapipe as mp
import pandas as pd
import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

class login_wd(QMainWindow):
    def __init__(self):
        super(login_wd, self).__init__()
        loadUi("AI.ui", self)

        # Kết nối sự kiện click của pushButton_2 với phương thức open_image
        self.pushButton_2.clicked.connect(self.open_image)
        print("PushButton_2 connected")

        # Kết nối sự kiện click của pushButton với phương thức run_processing
        self.pushButton.clicked.connect(self.run_processing)
        print("PushButton connected")

        # Tìm các QLabel có tên 'upload', 'anh1', 'anh2', 'anh3'
        self.label_image = self.findChild(QLabel, 'upload')
        self.label_anh1 = self.findChild(QLabel, 'anh1')
        self.label_anh2 = self.findChild(QLabel, 'anh2')
        self.label_anh3 = self.findChild(QLabel, 'anh3')
        self.label_anh = self.findChild(QLabel, 'anh')
        if not self.label_image:
            print("Label 'upload' not found")
        if not self.label_anh1:
            print("Label 'anh1' not found")
        if not self.label_anh2:
            print("Label 'anh2' not found")
        if not self.label_anh3:
            print("Label 'anh3' not found")

        # Biến lưu đường dẫn ảnh đã chọn
        self.selected_image_path = None  

        # Biến đếm số lần xuất hiện của từng nhãn
        self.label_counts = {
            '1': 0,
            '2': 0,
            '3': 0
        }

    def open_image(self):
        print("open_image triggered")
        # Mở hộp thoại chọn file và lấy đường dẫn file từ bất kỳ thư mục nào
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")

        # Nếu người dùng chọn một file, hiển thị nó và lưu đường dẫn
        if file_name:
            print(f"File selected: {file_name}")
            pixmap = QPixmap(file_name)
            self.label_image.setPixmap(pixmap)
            self.selected_image_path = file_name  # Lưu đường dẫn ảnh đã chọn
        else:
            print("No file selected")

    def run_processing(self):
        if not self.selected_image_path:
            print("No image selected. Please select an image first.")
            return
        
        print("run_processing triggered")
        # Gọi hàm xử lý ảnh với đường dẫn ảnh đã chọn
        self.process_image(self.selected_image_path)

    def process_image(self, image_path):
        # Khởi tạo Mediapipe Pose
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils

        # Đọc ảnh gốc
        image = cv.imread(image_path)

        # Sử dụng YOLO để nhận diện đối tượng trong ảnh
        yolo_model_path = 'E:\\AIII\\yolov9-pose.pt'
        model_yolo = YOLO(yolo_model_path)
        results = model_yolo.predict(image_path, conf=0.6)

        # Cắt các vùng chứa đối tượng nhận diện
        crops = {}
        for i, result in enumerate(results[0].boxes.xyxy.tolist()):
            x1, y1, x2, y2 = map(int, result)
            crop = image[y1:y2, x1:x2]
            crops[f'Person {i + 1}:'] = crop

        def make_landmarks(results):
            """
            Trích xuất các điểm landmarks từ kết quả của Mediapipe Pose.
            """
            landmarks = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
            return np.array(landmarks)

        def predict_behavior(crop, pose_model, behavior_model):
            results = pose_model.process(crop)
            
            # Trích xuất các điểm landmarks
            landmarks = make_landmarks(results)
            
            # Kiểm tra nếu số lượng đặc trưng phù hợp với mô hình đã huấn luyện
            if landmarks.size == 65:  # Nếu landmarks có 65 điểm, thêm một điểm giả để đạt 66
                landmarks = np.append(landmarks, 0)
            
            # Dự đoán hành vi
            prediction = behavior_model.predict([landmarks])
            return prediction[0]  # Trả về nhãn hành vi dự đoán

        # Đọc dữ liệu từ tệp Excel
        data_path = 'E:\\AIII\\datav2.xlsx'
        data = pd.read_excel(data_path)

        # Chuẩn bị dữ liệu cho huấn luyện mô hình
        X = data.drop('Label', axis=1)
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình SVM với GridSearchCV
        log_reg_grid = {'C': np.logspace(-4, 4, 20)}
        gs_log_reg = GridSearchCV(SVC(), log_reg_grid, cv=5, verbose=True)
        gs_log_reg.fit(X_train, y_train)

        # Huấn luyện mô hình SVM với giá trị C tốt nhất
        model = SVC(C=gs_log_reg.best_params_['C'])
        model.fit(X_train, y_train)

        # Đếm số lần xuất hiện của từng nhãn
        self.label_counts = {key: 0 for key in self.label_counts}  # Reset counts
        anh11 = 0
        anh22 = 0
        anh33 = 0
        c = 0
        for key, crop in crops.items():
            # Sử dụng hàm dự đoán để xác định hành vi của đối tượng
            prediction = predict_behavior(crop, pose, model)
            if(prediction == 1):
                anh11 += 1
            elif (prediction == 2):
                anh22 += 1
            else :
                anh33 += 1
            # In kết quả dự đoán
            print(f"{key} dự đoán hành vi là: {prediction}")

            # Cập nhật số lần xuất hiện của từng nhãn
            if prediction in self.label_counts:
                self.label_counts[prediction] += 1
        c = anh11 + anh22 + anh33
        
        # Cập nhật QLabel với số lần xuất hiện của từng nhãn
        self.label_anh.setText(f" {c}")
        self.label_anh1.setText(f" {anh11}")
        self.label_anh2.setText(f" {anh22}")
        self.label_anh3.setText(f" {anh33}")

app = QApplication(sys.argv)
widget = QStackedWidget()

login_f = login_wd()

widget.addWidget(login_f)

widget.setCurrentIndex(0)
widget.setFixedWidth(977)
widget.setFixedHeight(700)
widget.show()

sys.exit(app.exec_())
