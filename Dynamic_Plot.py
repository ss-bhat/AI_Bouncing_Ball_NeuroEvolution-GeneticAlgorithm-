import matplotlib.pyplot as plt


class DynamicUpdate:
    """ Dynamic plotting to keep generation and fitness. Used to see if algo reached local/global maxima """
    min_x = 0
    max_x = 10

    def on_launch(self):

        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        self.ax.grid()

    def on_running(self, xdata, ydata):

        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)

        self.ax.relim()
        self.ax.autoscale_view()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

