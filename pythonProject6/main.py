import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

# --- Завантаження даних ---
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# --- Sigmoid ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Ініціалізація ---
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

# --- Пряме + зворотнє поширення ---
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost

# --- Оптимізація (градієнтний спуск) ---
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw, db = grads["dw"], grads["db"]
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")
    return {"w": w, "b": b}, {"dw": dw, "db": db}, costs

# --- Прогноз ---
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

# --- Модель ---
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w, b = parameters["w"], parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print(f"train accuracy: {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100:.2f} %")
    print(f"test accuracy: {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100:.2f} %")

    return {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

# --- Передбачення на нових зображеннях ---
def predict_and_show(image_path, model, num_px):
    image = Image.open(image_path)
    image = image.resize((num_px, num_px))
    image_np = np.asarray(image) / 255.
    image_flatten = image_np.reshape((num_px * num_px * 3, 1))

    prediction = predict(model["w"], model["b"], image_flatten)
    plt.imshow(image_np)
    plt.axis('off')
    title = f"Prediction: {'cat' if int(prediction) == 1 else 'non-cat'}"
    plt.title(title)
    plt.show()
    return image_path.split("/")[-1], int(prediction)

# --- Запуск ---
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Навчання моделі
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# --- Побудова графіка функції витрат ---
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# --- Тестування на власних зображеннях ---
image_paths = [
    "gargouille.jpg",
    "cat_in_iran.jpg",
    "my_image.jpg",
    "my_image2.jpg",
    "la_defense.jpg"
]

# Прогноз для кожного зображення
# Виклик функції без абсолютного шляху
for img_path in image_paths:
    try:
        name, prediction = predict_and_show(img_path, d, num_px)
        print(f"{name} → Prediction: {'cat' if prediction == 1 else 'non-cat'}")
    except Exception as e:
        print(f"Помилка з файлом {img_path}: {e}")