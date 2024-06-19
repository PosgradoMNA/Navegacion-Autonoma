"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time



#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 30

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))

# Function to display the captured image in a separate window
def display_image_opencv(image):
    # Display captured image
    cv2.imshow("Captured Image", image)
    
    # Filter image to search for yellow (replace this with your filtering code)
    # Assuming you have a function called "filter_yellow" to filter yellow color
    filtered_image = filter_yellow(image)
    
    # Display filtered image
    cv2.imshow("Filtered Image", filtered_image)
    
    # Wait briefly for a key press (1 millisecond)
    cv2.waitKey(1)


def filter_yellow(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Bitwise-AND mask and original image
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    
    return filtered_image

def locate_yellow_line(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour corresponding to the yellow line
    yellow_line_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10:  # Adjust this threshold as needed
            yellow_line_contour = contour
            break
    
    if yellow_line_contour is not None:
        # Find the centroid of the contour (horizontal position of the yellow line)
        M = cv2.moments(yellow_line_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            return centroid_x
        else:
            # Handle division by zero (e.g., return None)
            return None
    else:
        # Yellow line not found
        return None
    

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    set_speed(0.1)
    

   

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        grey_image = greyscale_cv2(image)

        display_image(display_img, grey_image)

        

         # Display captured image in a separate window
        display_image_opencv(image)
        posss=locate_yellow_line(image)
    

        
        # Read keyboard
        key=keyboard.getKey()
        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
        elif key == ord('Q'): #quit
            print("code_stop")
            break

        elif key == ord("W"):
            print("test")


        elif posss != None:
                       
            dif=35-posss
            hy=100
            ang= dif/hy
            print("Seno del ángulo:", ang)

            set_steering_angle(-1*ang)

        elif posss == None:
            ang = -0.15
            set_steering_angle(-1*ang)

            



            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

# Close OpenCV windows after the loop finishes
cv2.destroyAllWindows()


if __name__ == "__main__":
    main()