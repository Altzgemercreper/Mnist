import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Загрузка датасета MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация изображений
train_images = train_images / 255.0
test_images = test_images / 255.0

# Создание и обучение модели
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Получение точности модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Точность на тестовых данных:', test_acc)

# Вывод случайной картинки из тестового датасета
random_image_index = np.random.randint(0, len(test_images))
random_image = test_images[random_image_index]
random_label = test_labels[random_image_index]

plt.figure()
plt.imshow(random_image, cmap=plt.cm.binary)
plt.title(f'Распознавание: {random_label}')
plt.show()
