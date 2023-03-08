import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the function to generate the ball's path
def ball_path(t):
    x = np.cos(t)
    y = np.sin(t)
    return x, y

# Define a new ball_path function
def ball_path(t):
    x = np.cos(2*t)
    y = np.sin(3*t)
    return x, y

# Create the animation using the new function
ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True, ball_path=ball_path)


# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))

# Create the ball object
ball, = ax.plot([], [], 'ro', markersize=10)

# Define the animation function
def animate(frame):
    t = frame / 100.0
    x, y = ball_path(t)
    ball.set_data(x, y)
    return ball,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

# Save the animation to an mp4 file
ani.save('red_ball.mp4', writer='ffmpeg')
