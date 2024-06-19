from controller import Display, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2


# Obtener imagen de la cámara
def get_image(camera):
    # Captura la imagen raw de la cámara
    raw_image = camera.getImage()
    # Convierte la imagen raw en un arreglo numpy de 3 dimensiones (alto, ancho, canales)
    # Utiliza np.frombuffer() para interpretar los bytes raw como un arreglo numpy de enteros sin signo de 8 bits (uint8)
    # La forma del arreglo resultante se especifica utilizando la altura y la anchura de la imagen de la cámara
    # El arreglo tiene 4 canales (RGBA) debido a la imagen en color
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image


# Procesamiento de imagen
def greyscale_cv2(image):
    # Convertir la imagen a escala de grises utilizando la función cv2.cvtColor()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de suavizado Gaussiano para reducir el ruido en la imagen en escala de grises
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Detectar bordes en la imagen suavizada utilizando el detector de bordes de Canny
    edges = cv2.Canny(blurred_img, 50, 150)

    # Definir un área de interés (ROI) en la imagen
    height, width = gray_img.shape[:2]
    mask = np.zeros_like(edges)
    # Definir los vértices del polígono que representa el área de interés
    polygon = np.array([[(0, height), (10, 0.6 * height), (width - 10, 0.65 * height), (width, height)]], np.int32)
    # Rellenar el polígono con color blanco (255) en la máscara
    cv2.fillPoly(mask, [polygon], 255)
    # Aplicar la máscara a los bordes detectados para obtener solo los bordes dentro del área de interés
    masked_edges = cv2.bitwise_and(edges, mask)

    # Definir los parámetros para la transformada de Hough
    rho = 0.5
    theta = np.pi / 180
    threshold = 50
    min_line_length = 100
    max_line_gap = 50

    # Aplicar la transformada de Hough para detectar líneas en la imagen con bordes
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Dibujar las líneas detectadas sobre una imagen en blanco
    line_image = np.zeros_like(gray_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Dibujar una línea sobre la imagen en blanco
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Combinar la imagen en escala de grises con los bordes detectados y las líneas dibujadas
    img_gray_with_edges = cv2.bitwise_or(masked_edges, line_image)
    # Alternativamente, se puede usar cv2.addWeighted() para combinar las imágenes con ponderaciones
    # result = cv2.addWeighted(masked_edges, 0.5, line_image, 0.5, 0)
    # O simplemente sumar las imágenes para obtener la misma combinación
    result = cv2.add(masked_edges, line_image)
    return img_gray_with_edges


# Display image
def display_image(display, image):
    # Convertir la imagen en escala de grises en una imagen RGB replicando el mismo canal en los tres canales de color
    image_rgb = np.dstack((image, image, image,))
    # Crear una nueva imagen utilizando la instancia de display.imageNew() con los datos de la imagen RGB
    image_ref = display.imageNew(
        image_rgb.tobytes(),  # Convertir la imagen en un búfer de bytes
        Display.RGB,          # Indicar que la imagen es en formato RGB
        width=image_rgb.shape[1],  # Ancho de la imagen
        height=image_rgb.shape[0],  # Altura de la imagen
    )
    # Pegar la imagen en el display en la posición (0, 0) con la opción de borrar el contenido anterior (False)
    display.imagePaste(image_ref, 0, 0, False)


# Velocidad de crucero constante del vehículo
speed = 60


# main
def main():
    # Crear una instancia del vehículo
    robot = Car()
    driver = Driver()

    # Obtener el paso de tiempo del mundo actual
    timestep = int(robot.getBasicTimeStep())

    # Crear una instancia de la cámara del vehículo
    camera = robot.getDevice("camera")
    # Habilitar la cámara con el paso de tiempo especificado
    camera.enable(timestep)

    # Crear una instancia para mostrar el procesamiento de la imagen
    display_img = Display("display_image")

    # Bucle principal que se ejecuta mientras la simulación esté activa
    while robot.step() != -1:
        # Obtener la imagen actual de la cámara
        image = get_image(camera)

        # Procesar y mostrar la imagen obtenida de la cámara
        processed_image = greyscale_cv2(image)
        display_image(display_img, processed_image)

        # Calcular el ángulo de dirección basado en la posición de las líneas detectadas
        steering_angle = calculate_steering_angle(processed_image)

        # Establecer el ángulo de dirección y la velocidad de crucero para el vehículo
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)


def calculate_steering_angle(processed_image):
    # Encontrar los contornos en la imagen procesada
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar variables para almacenar las posiciones de los carriles izquierdo y derecho
    left_lane_position = None
    right_lane_position = None

    # Iterar a través de los contornos detectados
    for contour in contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Verificar si el contorno es convexo y tiene suficientes puntos
        if cv2.isContourConvex(approx) and len(approx) >= 4:
            # Encontrar el rectángulo delimitador del contorno
            x, y, w, h = cv2.boundingRect(contour)

            # Calcular el centroide del rectángulo delimitador
            centroid_x = x + w // 2

            # Actualizar las posiciones de los carriles izquierdo y derecho basadas en la posición del centroide
            if left_lane_position is None or centroid_x < left_lane_position:
                left_lane_position = centroid_x
            if right_lane_position is None or centroid_x > right_lane_position:
                right_lane_position = centroid_x

    # Calcular el punto medio entre las posiciones de los carriles izquierdo y derecho
    if left_lane_position is not None and right_lane_position is not None:
        midpoint = (left_lane_position + right_lane_position) / 2

        # Calcular el ángulo basado en la posición del punto medio
        image_width = processed_image.shape[1]
        angle = (midpoint - image_width / 2) / (image_width / 2) * 0.5
        return angle

    # Si no se detectan carriles, girar a la derecha (ángulo positivo)
    return 0.2


if __name__ == "__main__":
    main()

