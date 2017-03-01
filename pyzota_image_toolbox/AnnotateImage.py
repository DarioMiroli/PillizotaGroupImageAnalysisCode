import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self, color='blue'):
        self.color = color
        self.ax = plt.gca()
        self.rects = []
        self.xs = []
        self.ys = []
        self.rectID = 0
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.pressed = False

    def on_press(self, event):
        if not self.pressed:
            self.pressed = True
            self.rects.append(Rectangle((0.1,0.1), 1, 1, alpha=0.5, color = self.color))
            self.xs.append([0,0])
            self.ys.append([0,0])
            self.xs[self.rectID][0] = event.xdata
            self.ys[self.rectID][0] = event.ydata
            self.ax.add_patch(self.rects[self.rectID])


    def on_motion(self,event):
        if self.pressed == True and event.xdata != None and event.ydata!= None:
            self.xs[self.rectID][1] = event.xdata
            self.ys[self.rectID][1] = event.ydata
            self.rects[self.rectID].set_xy((self.xs[self.rectID][0], self.ys[self.rectID][0]))
            self.rects[self.rectID].set_width(self.xs[self.rectID][1] - self.xs[self.rectID][0])
            self.rects[self.rectID].set_height(self.ys[self.rectID][1] - self.ys[self.rectID][0])
            self.ax.figure.canvas.draw()


    def on_release(self, event):
        if self.pressed == True and event.xdata != None and event.ydata!= None:
            self.pressed = False
            self.xs[self.rectID][1] = event.xdata
            self.ys[self.rectID][1] = event.ydata
            self.rects[self.rectID].set_xy((self.xs[self.rectID][0], self.ys[self.rectID][0]))
            self.rects[self.rectID].set_width(self.xs[self.rectID][1] - self.xs[self.rectID][0])
            self.rects[self.rectID].set_height(self.ys[self.rectID][1] - self.ys[self.rectID][0])
            self.ax.figure.canvas.draw()
            self.rectID = self.rectID + 1

    def getRects(self):
        recArray = []
        for rec in self.rects:
            x1 = rec.get_x()
            y1 = rec.get_y()
            x2 = x1 + rec.get_width()
            y2 = y1 + rec.get_height()
            recArray.append([x1,x2,y1,y2])
        return recArray
