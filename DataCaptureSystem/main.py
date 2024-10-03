
from process_streaming_recognize import UartListener, DataProcessor, DequeFIFO, MatrixFIFO
from queue import Queue, Empty
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import time
import sys
import subprocess
# -----------------------------------------------
# from app_layout import Ui_MainWindow
from ui.appUI import Ui_MainWindow

def update_figure():
    try:
        # 首先尝试非阻塞地获取所有数据
        data_available = False
        try:
            r1_data = R1_Data.get_nowait()
            data_available = True
        except Empty:
            r1_data = None  # 或者设定为默认值
        
        try:
            r3_data = R3_Data.get_nowait()
            data_available = True
        except Empty:
            r3_data = None  # 或者设定为默认值
        
        try:
            r4_data = R4_Data.get_nowait()
            data_available = True
        except Empty:
            r4_data = None  # 或者设定为默认值


        # 检查是否至少获得了一个数据队列的内容
        if not data_available:
            raise Empty("All queues are empty")


        if r1_data is not None: 
            x_data = range(len(r1_data))
            writexaxisView.setData(x_data, r1_data[:,0])
            writezaxisView.setData(x_data, r1_data[:,1])
            
            # 如果合上双手，则更新图形界面
            if r1_data[-1,2]==1:
                if len(r1_data)<256:
                    writeTrajView.setData(r1_data[:,0], r1_data[:,1], pen=pg.mkPen(color=(0, 0, 0), width=8))
                else:
                    writeTrajView.setData(r1_data[-256:,0], r1_data[-256:,1], pen=pg.mkPen(color=(0, 0, 0), width=8))
            else:
                # pass
                writeTrajView.setData([0], [0], pen=pg.mkPen(color=(100, 0, 0), width=8))

        imgUdopplerTime.setImage(r3_data, levels=[0, 0.01])
        imgRangeTime.setImage(r4_data, levels=[0, 0.05])
        QtCore.QTimer.singleShot(1, update_figure)
    except Empty as e:
        # 当所有队列都空时，处理异常
        # print(e)
        QtCore.QTimer.singleShot(100, update_figure)

def openradar(logwidget):
    global collector, processor # 用qt线程一定要全局化
    printlog(logwidget, string='start!', fontcolor='green')
    collector = UartListener('Listener', indexPosition, matrix_fifo)
    processor = DataProcessor('Processor', indexPosition, matrix_fifo, R1_Data, R3_Data, R4_Data)
    processor.sendLabel.connect(lambda d, logwidget=logwidget: printlogdict(logwidget, d))
    processor.setlabelinit()
    update_figure()
    collector.start()
    processor.start()
    
def printlogdict(logwidget, d):
    string = d["string"]
    fontcolor = d["fontcolor"]
    # print(status)
    printlog(logwidget, string, fontcolor)

def printlog(logwidget, string, fontcolor):
    logwidget.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    logwidget.append("<font color="+fontcolor+">" +
                  str(gettime)+"-->"+string+"</font>")

def colorMapSet():
        # Colormap
    position = np.arange(64)
    position = position / 64
    position[0] = 0
    position = np.flip(position)
    colors = [[62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255], [67, 50, 202, 255], [69, 55, 213, 255],
              [70, 60, 222, 255], [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255], [69, 82, 240, 255],
              [68, 88, 243, 255],
              [68, 94, 247, 255], [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255], [56, 117, 254, 255],
              [50, 123, 252, 255],
              [47, 129, 250, 255], [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255], [39, 150, 235, 255],
              [37, 155, 232, 255],
              [35, 160, 229, 255], [31, 164, 225, 255], [28, 129, 222, 255], [24, 173, 219, 255], [17, 177, 214, 255],
              [7, 181, 208, 255],
              [1, 184, 202, 255], [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255], [36, 193, 174, 255],
              [44, 195, 167, 255],
              [49, 198, 159, 255], [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255], [88, 202, 121, 255],
              [102, 202, 111, 255],
              [116, 201, 100, 255], [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255], [171, 199, 57, 255],
              [185, 196, 49, 255],
              [197, 194, 42, 255], [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255], [239, 186, 53, 255],
              [248, 186, 61, 255],
              [254, 189, 60, 255], [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255], [248, 214, 46, 255],
              [246, 220, 43, 255],
              [245, 227, 39, 255], [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255], [249, 251, 20, 255]]
    colors = np.flip(colors, axis=0)
    color_map = pg.ColorMap(position, colors)
    lookup_table = color_map.getLookupTable(0.0, 1.0, 256)
    return lookup_table

def application():
    global writeTrajView, writexaxisView, writezaxisView, imgUdopplerTime, imgAngleTime, imgRangeTime
    # ---------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    writeTrajView = ui.plot_writeTraj_leap
    writexaxisView = ui.plot_writexTime_leap
    writezaxisView = ui.plot_writezTime_leap

    # 改了D:\Applications\anaconda3\Lib\site-packages\pyqtgraph\graphicsItems\ViewBox
    # 里的ViewBox.py第939行padding = self.suggestPadding(ax)改成padding = 0

    udopplerView = ui.graphicsView_3.addViewBox()
    ui.graphicsView_3.setCentralWidget(udopplerView)#去边界
    RangeView = ui.graphicsView_4.addViewBox()
    ui.graphicsView_4.setCentralWidget(RangeView)#去边界
    AngleView = ui.graphicsView_5.addViewBox()
    ui.graphicsView_5.setCentralWidget(AngleView)#去边界

    starbtn = ui.action
    logwidget = ui.printLog
    # lock the aspect ratio so pixels are always square
    # view_rdi.setAspectLocked(True)
    # view_rai.setAspectLocked(True)

    imgUdopplerTime = pg.ImageItem(border=None)
    imgAngleTime = pg.ImageItem(border=None)
    imgRangeTime = pg.ImageItem(border=None)
    lookup_table = colorMapSet()
    imgUdopplerTime.setLookupTable(lookup_table)
    imgAngleTime.setLookupTable(lookup_table)
    imgRangeTime.setLookupTable(lookup_table)

    udopplerView.addItem(imgUdopplerTime)
    AngleView.addItem(imgAngleTime)
    RangeView.addItem(imgRangeTime)

    starbtn.triggered.connect(lambda:openradar(logwidget))
    app.instance().exec_()

if __name__ == '__main__':
    # Queue for access data
    R1_Data = Queue()
    R3_Data = Queue()
    R4_Data = Queue()
    # is_draw = Queue()
    # 创建FIFO实例
    indexPosition = DequeFIFO(n=64)
    matrix_fifo = MatrixFIFO(num_chirps=64, num_adcsamples=32, max_num_chirps=1024)

    # 指定.exe文件的路径
    exe_path = 'D:/Applications/Ultraleap/TrackingService/bin/LeapSvc.exe'

    # 使用subprocess.Popen启动程序
    process = subprocess.Popen(exe_path)

    application()
    # 等待进程完成
    process.wait()
    sys.exit()
