import matplotlib.pyplot as plt
import numpy as np



def ball_path(t):
    x = np.cos(2*t)
    y = np.sin(3*t)
    return x, y


def object_position(x0, y0, v, u, t_max, dt=0.1):
    # Generate time array
    time_array = [i * dt for i in range(int(t_max/dt) + 1)]
    
    # Generate x and y positions for each time in time array
    coordinates = [ball_path(t) for t in time_array]
    x_pos, y_pos = zip(*coordinates)
    # x_pos = [v * t + x0 for t in time_array]
    # y_pos = [u * t + y0 for t in time_array]

    # Plot the x and y positions in the x-y plane
    plt.scatter(x_pos, y_pos)
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    plt.title('Object position in x-y plane')
    plt.show()
    return list(x_pos), list(y_pos)

x, y = object_position(0, 0, 3, 4, 20)

data = np.array(list(zip(x, y)), dtype="float32")
plt.scatter(X_train[:, 0], X_train[:, 1], label='Original data')
plt.scatter(X_test[:, 0], X_test[:, 1], label='Test data')

plt.show()






import numpy as np
from sklearn.model_selection import train_test_split

# Generate sample data
# data = np.random.rand(100, 2)
data = np.array(list(zip(x, y)), dtype="float32")

# Define window size
window_size = 4

# Prepare input and output data
X = []
y = []
for i in range(len(data) - window_size):
    X.append(data[i:i+window_size])
    y.append(data[i+window_size])

# Convert input and output data to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,shuffle = False,)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1,shuffle = False,)

# Print shapes of input and output data for each set
print("Train shapes: X = {}, y = {}".format(X_train.shape, y_train.shape))
print("Validation shapes: X = {}, y = {}".format(X_val.shape, y_val.shape))
print("Test shapes: X = {}, y = {}".format(X_test.shape, y_test.shape))

























from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define window size
window_size = 4

# Define LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(window_size, 2)))
model.add(Dense(2, activation='linear'))

# Compile model
model.compile(loss='mse', optimizer='adam')

# Print model summary
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss = model.evaluate(X_test, y_test)

# Make predictions
y_pred = model.predict(X_test)









plt.scatter(data[:, 0], data[:, 1], label='Original data')
# plt.scatter(X_test[:, 0], X_test[:, 1], label='Original data')
plt.scatter(y_pred[:, 0], y_pred[:, 1], label='Predicted points')
plt.legend()
plt.show()




y_test_flat = y_test.reshape(-1, 1)
y_pred_flat = y_pred.reshape(-1, 1)

# plot the original and predicted data
plt.plot(y_test_flat, label='Original')
plt.plot(y_pred_flat, label='Predicted')
plt.legend()
plt.show()