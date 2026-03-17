import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        # Коректні точки для рівностороннього трикутника
        points = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
        cv2.fillPoly(img, [points], color)
    return img


X = []
y = []
colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}
shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for i in range(20):  # Збільшимо вибірку для стабільності
            img = generate_image(bgr, shape)

            # Додаємо площу (кількість не чорних пікселів) як ознаку для форми
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            area = np.sum(gray > 0)

            mean_color = cv2.mean(img)[:3]
            # Тепер у нас 4 ознаки: R, G, B та Area
            features = [mean_color[0], mean_color[1], mean_color[2], area]

            X.append(features)
            y.append(f'{color_name}_{shape}')

# Виправлено: stratify замість startify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print(f'Точність: {round(model.score(X_test, y_test) * 100, 2)}%')

# Тестування
test_color = (0, 255, 0)  # Зелений
test_shape = "circle"
test_img = generate_image(test_color, test_shape)

area_test = np.sum(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) > 0)
mean_color_test = cv2.mean(test_img)[:3]
features_test = [mean_color_test[0], mean_color_test[1], mean_color_test[2], area_test]

prediction = model.predict([features_test])
print(f'Передбачення: {prediction[0]}')

cv2.imshow('Test Image', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
