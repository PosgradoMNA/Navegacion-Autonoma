"""camera_pid controller."""

from controller import Display, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Image processing
def greyscale_cv2(image):
    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de suavizado Gaussiano para reducir el ruido
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Detectar bordes
    edges = cv2.Canny(blurred_img, 50, 150)
    
    # Definir un área de interés (ROI)
    height, width = gray_img.shape[:2]
    mask = np.zeros_like(edges)
    #polygon = np.array([[(0, height),(20, 0.6 * height),(width-20, 0.65 * height),(width, height)]], np.int32)
    polygon = np.array([[(0, height),(10, 0.6 * height),(width-10, 0.65 * height),(width, height)]], np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Definir los parámetros para la transformada de Hough
    rho = 0.5
    theta = np.pi / 180
    threshold = 50
    min_line_length = 100
    max_line_gap = 50
    
    # Aplicar la transformada de Hough para detectar las líneas en la imagen
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap) 
    
    # Dibujar las líneas detectadas sobre la imagen original
    line_image = np.zeros_like(gray_img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Combinar la imagen en escala de grises con los bordes detectados
    #img_gray_with_edges = cv2.bitwise_or(gray_img, edges)
    img_gray_with_edges = cv2.bitwise_or(masked_edges, line_image)
    #result = cv2.addWeighted(masked_edges, 0.5, line_image, 0.5, 0)
    result = cv2.add(masked_edges, line_image)
    return img_gray_with_edges
# Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image, image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Set a constant cruising speed
speed = 45

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Processing display
    display_img = Display("display_image")

    # Loop until the simulation is stopped
    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        processed_image = greyscale_cv2(image)
        display_image(display_img, processed_image)

        # Calculate steering angle based on the position of the lines
        steering_angle = calculate_steering_angle(processed_image)

        # Set steering angle and speed for the vehicle
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(speed)

def calculate_steering_angle(processed_image):
    # Find contours in the processed image
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store left and right lane positions
    left_lane_position = None
    right_lane_position = None

    # Iterate through contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the contour is convex and has enough points
        if cv2.isContourConvex(approx) and len(approx) >= 4:
            # Find the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate the centroid of the bounding rectangle
            centroid_x = x + w // 2
            
            # Update left and right lane positions based on centroid position
            if left_lane_position is None or centroid_x < left_lane_position:
                left_lane_position = centroid_x
            if right_lane_position is None or centroid_x > right_lane_position:
                right_lane_position = centroid_x
    
    # Calculate the midpoint between the left and right lane positions
    if left_lane_position is not None and right_lane_position is not None:
        midpoint = (left_lane_position + right_lane_position) / 2
        
        # Calculate the angle based on the midpoint position
        image_width = processed_image.shape[1]
        angle = (midpoint - image_width / 2) / (image_width / 2) * 0.5
        return angle

    # If no lanes detected, turn right
    return -0.5

if __name__ == "__main__":
    main()
