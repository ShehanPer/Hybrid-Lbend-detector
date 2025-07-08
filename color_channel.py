import cv2

cap = cv2.VideoCapture(2)  # Change to your camera index
if not cap.isOpened():  
    print("Error: Could not open video stream.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow('Camera Feed', frame)
    #red channel
    red_channel = frame[:, :, 2]  # Extract the red channel
    cv2.imshow('Red Channel', red_channel)
    # green channel
    green_channel = frame[:, :, 1]  # Extract the green channel
    cv2.imshow('Green Channel', green_channel)
    # blue channel
    blue_channel = frame[:, :, 0]  # Extract the blue channel
    cv2.imshow('Blue Channel', blue_channel)


    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break