# Circle Tracking with speed and acceleration predictions and the re-tracking mechanism
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import argparse

# Speed and acceleration predictions of the next position based on the last n positions
def predict_position_speed_acceleration(positions, n, frame_width, frame_height, acceleration_weight=0):
    if len(positions) <= 2:
        return positions[-1][0], positions[-1][1]  # Use current position if insufficient history

    if len(positions) < n:
        n = len(positions) - 1  # Adjust n to the available history

    # Calculate speed in x and y
    x_diff = [positions[i][0] - positions[i - 1][0] for i in range(-1, -n, -1)]
    y_diff = [positions[i][1] - positions[i - 1][1] for i in range(-1, -n, -1)]

    # Calculate average speed in x and y
    avg_x_diff = sum(x_diff) / n
    avg_y_diff = sum(y_diff) / n

    # Calculate average acceleration in x and y directions
    accel_x = sum(x_diff[i] - x_diff[i - 1] for i in range(1, n - 1)) / (n - 2) if n > 2 else 0
    accel_y = sum(y_diff[i] - y_diff[i - 1] for i in range(1, n - 1)) / (n - 2) if n > 2 else 0

    # Extrapolate next position based on average change and acceleration
    next_x = int(positions[-1][0] + avg_x_diff + acceleration_weight * accel_x)
    next_y = int(positions[-1][1] + avg_y_diff + acceleration_weight * accel_y)

    # Ensure predicted coordinates stay within the frame
    next_x = max(0, min(next_x, frame_width - 1))
    next_y = max(0, min(next_y, frame_height - 1))

    return next_x, next_y

def calculate_median_gray_value(frame, x, y, r):
    # Bounding box around the circle
    bbox = (x - r, y - r, 2 * r, 2 * r)

    # Get the region of interest (ROI) using the bounding box
    roi = frame[y - r:y + r, x - r:x + r]

    # Calculate median grayscale value within the ROI
    median_gray_value = np.median(roi)

    return median_gray_value

def initialize_tracker(frame, x, y, r):
    # Calculate median grayscale value within the circle
    median_gray_value = calculate_median_gray_value(frame, x, y, r)

    # Initialize tracker for each detected circle
    tracker = {
        'positions': [(x, y)],
        'age': 0,
        'radius': r,
        'gray_scale': median_gray_value,
        'color': (np.random.randint(200, 256), np.random.randint(0, 200), np.random.randint(100, 256))
        }
    return tracker

def CircaTrack(video_path, output_video_path, output_txt_path):
    # Parameters and variables
    n_frames = 3  # Number of previous frames for calculating the speed and acceleration, minimum 2 for speed and 3 for acceleration
    acceleration_weight = 0  # Acceleration weight for the amount of contribution of acceleration, from 0 as no contribution to 1 as full contribution
    threshold_age = 20  # Threshold for age to loss tracking
    threshold_age_reassign = 3  # Threshold of age to reassign unselected circles
    threshold_distance = 200  # Threshold for distance between predicted and detected circles
    threshold_radius = 5  # Threshold for the radius difference between current and detected circles
    threshold_gray = 10  # Threshold for the gray scale difference between current and detected circles

    dp = 0.5
    minDist = 35
    param1 = 15
    param2 = 30
    kernel_size = (7, 7)

    # Initialize trackers dictionary to store trackers for circles
    trackers = {}

    n_initiated_trackers = 0  # Number of initiated trackers
    target_count = 10  # Set the total number of bubbles to track

    circle_positions = {}  # Dictionary to store circle positions per frame

    frame_number = 0

    # Initialize video capture
    video = cv2.VideoCapture(video_path)

    # Output video settings
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        # If video reaches the end, warn and break the loop
        if not ret:
            print('Failed to read video')
            break

        frame_number += 1  # Increment frame number
        #print(f'frame number: {frame_number}')

        # Convert frame to grayscale for circle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, kernel_size, 0)

        # Detect circles using HoughCircles on the thresholded image
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                   param1=param1, param2=param2, minRadius=20, maxRadius=150)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            detected_circles = []
            #print(f'Number of detected circles: {len(circles)}')

            # Initialize trackers
            if frame_number == 1:
                for i, (x, y, r) in enumerate(circles):
                    if n_initiated_trackers < target_count:
                        # Initiate the tracker for the detected circle
                        trackers[i] = initialize_tracker(frame, x, y, r)
                        n_initiated_trackers += 1

            else:
                # Update each circle in trackers
                for circle_id, data in trackers.items():
                    # Predict x, y coordinate for the current frame based on history
                    predicted_x, predicted_y = predict_position_speed_acceleration(data['positions'], n_frames, frame_width, frame_height, acceleration_weight)

                    number_circle_checked = 0
                    # Loop over detected circles
                    for i, (x, y, r) in enumerate(circles):
                        # Calculate median grayscale value within the circle
                        circle_gray_value = calculate_median_gray_value(frame, x, y, r)
                        # Calculate the distance between x, y coordinates of the circle and the prediction
                        distance = np.sqrt((predicted_x - x) ** 2 + (predicted_y - y) ** 2)
                        # print(f'predicted_x {predicted_x}, x {x}, predicted_y {predicted_y}, y {y}, distanct {distance}')
                        # Check if the detected circle is around the predicted position
                        if distance < threshold_distance and abs(data['radius'] - r) < threshold_radius and abs(data['gray_scale'] - circle_gray_value) < threshold_gray:
                            # Update tracker with new coordinates
                            data['positions'].append((x, y))  # Append new position
                            data['age'] = 0
                            detected_circles.append(i)
                            break

                        number_circle_checked += 1

                    if number_circle_checked == len(circles):
                        # Assign predicted x and y as the new location if no detected circle is around
                        # The circle might move at the same speed
                        # Assigning predicted xy might help with finding circles in the following frames
                        # print(f'use predicted position, number of circles {len(circles)} number circle checked {number_circle_checked}')
                        data['positions'].append((predicted_x, predicted_y))
                        data['age'] += 1

                # Deal with unselected circles
                for circle_id, (x, y, r) in enumerate(circles):
                    if circle_id not in detected_circles:
                        # If the target count of initialized circles is not reached, initialize the tracker for this circle
                        if n_initiated_trackers < target_count:
                            trackers[len(trackers)] = initialize_tracker(frame, x, y, r)
                            n_initiated_trackers += 1
                        else:
                            # Reassign the unselected circle to one of the similar radius and gray scale trackers that are undetected for a threshold age
                            # Calculate median grayscale value within the circle
                            circle_gray_value = calculate_median_gray_value(frame, x, y, r)
                            for circle_id, data in trackers.items():
                                if data['age'] >= threshold_age_reassign and abs(data['radius'] - r) < threshold_radius and abs(data['gray_scale'] - circle_gray_value) < threshold_gray:
                                    data['positions'] = [(x, y)]  # Restart tracking the position
                                    data['age'] = 0
                                    break

        # Draw boxes for circles with age smaller than threshold
        for circle_id, data in trackers.items():
            if data['age'] < threshold_age:
                x, y = data['positions'][-1]  # Last position for drawing
                color = data['color']
                cv2.rectangle(frame, (x - data['radius'], y - data['radius']), (x + data['radius'], y + data['radius']), color, 2)
                cv2.putText(frame, f'{circle_id + 1}', (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Store circle positions with age less than threshold for this frame
        circle_positions[frame_number] = [data['positions'][-1] if len(data['positions']) > 0 and data['age'] < threshold_age else (-1, -1) for _, data in trackers.items()]

        # Write frame with circles to output video
        output_video.write(frame)

        # Display the frame (optional)
        #cv2_imshow(frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Prepare headers
    headers = ['Frame no.'] + [f'Bubble_{i}' for i in range(1, 11)]

    # Write circle positions to a text file
    with open(output_txt_path, "w") as file:
        # Write headers
        file.write("\t".join(headers) + "\n")

        # Write circle positions data
        for frame_num, circles in circle_positions.items():
            line = f"{frame_num}"
            for circle in circles:
                line += f"\t({circle[0]}, {circle[1]})"
            file.write(line + "\n")

    # Release resources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()



def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Circle Tracking with Speed and Acceleration Predictions')

    # Add command line arguments
    parser.add_argument('--video_path', type=str, help='Path to the input video')
    parser.add_argument('--output_video_path', type=str, help='Path to save the output video')
    parser.add_argument('--output_txt_path', type=str, help='Path to save the output circle positions text file')

    # Parse the arguments
    args = parser.parse_args()

    # Call CircaTrack with the provided arguments
    CircaTrack(args.video_path, args.output_video_path, args.output_txt_path)

if __name__ == "__main__":
    main()