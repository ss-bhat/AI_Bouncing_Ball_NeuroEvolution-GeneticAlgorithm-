
""" Credit: DJ Oamen – Bouncing ball game code python.
            https://www.youtube.com/watch?v=9tVCYIcwNjw"""


from tkinter import *
import time
import random
import numpy as np


class Ball:

    def __init__(self, canvas, paddle, color):
        """ Canvas parameter to the ball. """
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        self.canvas.move(self.id, 245, 100)
        starts = [-3, -2, -1, 1, 2, 3]
        random.shuffle(starts)
        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False
        self.score = 0

    def hit_paddle(self, pos):
        """ Check if the ball hits the paddle """
        paddle_pos = self.canvas.coords(self.paddle.id)

        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:
                self.score += 1
                return True

        return False

    def draw(self):
        """ Draw the Ball"""
        self.canvas.move(self.id, self.x, self.y)
        pos = self.canvas.coords(self.id)

        if pos[1] <= 0:
            self.y = 2
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if self.hit_paddle(pos) is True:
            self.y = -2
        if pos[0] <= 0:
            self.x = 2
        if pos[2] >= self.canvas_width:
            self.x = -2


class Paddle:

    def __init__(self, canvas, color):
        """ Canvas parameters """
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 390)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)

    def draw(self):
        """ Draw the paddle """
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)

        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self, evt):
        """ Turn left method"""
        pos = self.canvas.coords(self.id)

        if pos[0] > 0:
            self.x = -3

    def turn_right(self, evt):
        """ Turn Right method """
        pos = self.canvas.coords(self.id)

        if pos[0] < 400:
            self.x = 3


def main(agent, gen, agt_no):
    """ Main function which generate the canvas window"""
    tk = Tk()
    quote = "Generation: "+str(gen)
    agt = "Agent: "+str(agt_no)+" out of 10"
    tk.title("AI Bouncing Ball Game - 17230755")
    tk.resizable(0, 0)
    tk.wm_attributes("-topmost", 1)
    canvas = Canvas(tk, width=500, height=400, bd=0, highlightthickness=0)
    canvas.create_text(400, 20, fill="darkblue", font=("Times 20 italic bold", 12), text=quote+'\n'+agt)
    canvas.pack()
    tk.update()

    paddle = Paddle(canvas, 'blue')
    ball = Ball(canvas, paddle, 'gray')

    while not ball.hit_bottom:

        if not ball.hit_bottom:
            ball.draw()
            paddle.draw()

            # Keep track of position of ball and paddle
            cord_ball_x = ball.canvas.coords(ball.id)[0]
            cord_ball_y = ball.canvas.coords(ball.id)[1]
            cord_paddle_x = paddle.canvas.coords(paddle.id)[0]
            cord_paddle_y = paddle.canvas.coords(paddle.id)[1]

            # Neural network agent input to control the paddle
            if agent.feed_forward(np.asarray([cord_ball_x-cord_paddle_x, cord_ball_y-cord_paddle_y]) / sum(
                    [cord_ball_x-cord_paddle_x, cord_ball_y-cord_paddle_y])) <= 0.5:
                paddle.turn_left(None)
            elif agent.feed_forward(np.asarray([cord_ball_x-cord_paddle_x, cord_ball_y-cord_paddle_y]) / sum(
                    [cord_ball_x-cord_paddle_x, cord_ball_y-cord_paddle_y])) > 0.5:
                paddle.turn_right(None)

        # Update the TkInter window
        tk.update_idletasks()
        tk.update()
        time.sleep(0.009)

    tk.destroy()
    print("For Generation: {}, Agent: {}, Score: {}".format(gen, agt_no, ball.score))

    # Return Fitness value for the each agent and store value in agent.fitness
    return ball.score - abs(cord_paddle_x-cord_ball_x)/600
