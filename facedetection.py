from tkinter import *
from PyQt4 import QtGui,QtCore
import cv2
import tensorflow as tf
import pyttsx3

faceCascade = cv2.CascadeClassifier('I:/projects/py/Data/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window,self).__init__(parent)
        self.setGeometry(150,150,680,565)
        self.setWindowTitle('Cam')
        self.video = QtGui.QLabel('', self)
        self.video.setGeometry(20, 20, 640, 485)
        self.btn1 = QtGui.QPushButton('Start', self)
        self.btn1.setGeometry(50, 515, 100, 30)
        self.btn1.clicked.connect(self.Start)
        self.btn2 = QtGui.QPushButton('Face', self)
        self.btn2.setGeometry(400, 515, 100, 30)
        self.btn2.clicked.connect(self.faced)
        self.btn3 = QtGui.QPushButton('Scan', self)
        self.btn3.setGeometry(170, 515, 100, 30)
        self.btn3.clicked.connect(self.Stop)
        self.btn4 = QtGui.QPushButton('Exit', self)
        self.btn4.setGeometry(290, 515, 100, 30)
        self.btn4.clicked.connect(self.Exit)
        myPixmap = QtGui.QPixmap("./Data/camera.jpg")
        myScaledPixmap = myPixmap.scaled(self.video.size())
        self.video.setPixmap(myScaledPixmap)
        self.cap = cv2.VideoCapture(0)
        self.show()
    def Start(self):
        self.fps=30
        self.timer = QtCore.QTimer()
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.a=frame
        self.video.setPixmap(pix)
        self.timer.timeout.connect(self.Start)
        self.timer.start(1000. / self.fps)
    def Stop(self):
        fr = cv2.cvtColor(self.a, cv2.COLOR_BGR2RGB)
        cv2.imwrite("Scan1.jpg", fr)
        self.timer.stop()
        image_data = tf.gfile.FastGFile("./Scan1.jpg", 'rb').read()

        label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("./Data/output_labels.txt")]

        with tf.gfile.FastGFile("./Data/output_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor, \
                                   {'DecodeJpeg/contents:0': image_data})

            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
            #  for node_id in a:
            node_id = top_k[0]
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score >= 0.60:
                print(human_string)
                print(score)
                k = pyttsx3.init()
                rate = k.getProperty('rate')
                k.setProperty('rate', rate - 30)
                k.runAndWait()
                k.say('Hello' + human_string)
                k.runAndWait()

            else:
                k = pyttsx3.init()
                rate = k.getProperty('rate')
                k.setProperty('rate', rate - 30)
                k.runAndWait()
                k.say('Unknown Person')
                k.runAndWait()
                print("Image Does Not Match")
    def faced(self):
        self.fps = 30
        self.timer = QtCore.QTimer()
        ret, frame = self.cap.read()
        # ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video.setPixmap(pix)
        self.timer.timeout.connect(self.faced)
        self.timer.start(1000. / self.fps)
    def Exit(self):
        sys.exit()
app=QtGui.QApplication(sys.argv)
GUI=Window()
sys.exit(app.exec_())
