import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self, color='blue',ax = None, mode ='Rects',data=None):
        self.mode = mode
        if ax == None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        if mode == 'Rmv':
            self.data = data
        self.color = color
        self.rects = []
        self.xs = []
        self.ys = []
        self.rectID = 0
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('key_press_event',self.on_key_press)
        self.pressed = False
        self.patches = []
        self.goodStart = False

    def on_press(self, event):
        if self.mode == 'Rmv':
            title = self.ax.get_title()
            print(title)
            self.ax.cla()
            self.ax.figure.canvas.draw_idle()
            x,y = int(event.xdata) , int(event.ydata)
            value = self.data[y][x]
            self.data[self.data==value] = -1
            self.ax.imshow(self.data, cmap= "jet",interpolation='None')
            self.ax.figure.canvas.draw_idle()
            self.ax.set_title(title)

        if self.mode == 'Rects':
            if not self.pressed:
                self.pressed = True
                self.goodStart = False
                if event.xdata != None and event.ydata !=None:
                    self.goodStart = True
                    self.rects.append(Rectangle((0.1,0.1), 1, 1, alpha=0.5, color = self.color))
                    self.xs.append([0,0])
                    self.ys.append([0,0])
                    self.xs[self.rectID][0] = event.xdata
                    self.ys[self.rectID][0] = event.ydata
                    patch = self.rects[self.rectID]
                    self.patches.append(patch)
                    self.ax.add_patch(patch)


    def on_motion(self,event):
        if self.pressed and event.xdata != None and event.ydata!= None and self.goodStart:
            self.xs[self.rectID][1] = event.xdata
            self.ys[self.rectID][1] = event.ydata
            self.rects[self.rectID].set_xy((self.xs[self.rectID][0], self.ys[self.rectID][0]))
            self.rects[self.rectID].set_width(self.xs[self.rectID][1] - self.xs[self.rectID][0])
            self.rects[self.rectID].set_height(self.ys[self.rectID][1] - self.ys[self.rectID][0])
            self.ax.figure.canvas.draw()


    def on_release(self, event):
        if event.xdata != None and event.ydata!= None and self.goodStart:
            self.xs[self.rectID][1] = event.xdata
            self.ys[self.rectID][1] = event.ydata
            self.rects[self.rectID].set_xy((self.xs[self.rectID][0], self.ys[self.rectID][0]))
            self.rects[self.rectID].set_width(self.xs[self.rectID][1] - self.xs[self.rectID][0])
            self.rects[self.rectID].set_height(self.ys[self.rectID][1] - self.ys[self.rectID][0])
            self.ax.figure.canvas.draw()
            self.rectID = self.rectID + 1
            self.pressed = False


    def on_key_press(self,event):
        if event.key == 'z' and not self.pressed:
            if len(self.patches) > 0:
                self.patches[-1].remove()
                del self.patches[-1]
                del self.rects[-1]
                self.rectID = self.rectID -1
                self.ax.figure.canvas.draw()

    def getRects(self):
        recArray = []
        for rec in self.rects:
            x1 = rec.get_x()
            y1 = rec.get_y()
            x2 = x1 + rec.get_width()
            y2 = y1 + rec.get_height()
            recArray.append([x1,x2,y1,y2])
        return recArray

    def getData(self):
        return self.data
