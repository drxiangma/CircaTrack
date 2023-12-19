import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from scipy.stats import mode
import argparse

def choose_tracker(tracker_type):
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()

def CircleTracking(tracker_type, input_video_path, output_video_path, output_txt_path):
    # Open the video file
    video = cv2.VideoCapture(input_video_path)

    # Read first frame
    success, frame = video.read()
    if not success:
        print('Failed to read video')

    # Output video settings
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps, (frame_width, frame_height))

    # Create an empty dictionary to store circle positions per frame
    circle_positions = {}

    frame_number = 0  # Initialize frame number

    # Create a dictionary to hold trackers for each circle
    trackers = {}

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1  # Increment frame number

        # Convert frame to grayscale for circle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using HoughCircles on the thresholded image, HP: 0.7, 50, 20, 40, 1, 300
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=0.7, minDist=50,
                                   param1=20, param2=40, minRadius=1, maxRadius=300)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            #print(f'Number of detected circles: {len(circles)}')

            # Loop over detected circles
            for circle_id, (x, y, r) in enumerate(circles):
                # Assign different colors to each circle based on circle_id using HSV color space
                hue = int(180 * circle_id / len(circles))  # Varying hue values
                color = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])

                # Bounding box around the circle
                bbox = (x - r, y - r, 2 * r, 2 * r)

                # Ensure the circle coordinates are within the image boundaries
                x = max(0, x - r)
                y = max(0, y - r)

                # Get the grayscale ROI of the detected circle
                circle_roi_gray = gray[y:y+2*r, x:x+2*r]

                # Calculate the mode of grayscale values within the circle's ROI
                mode_result = mode(circle_roi_gray.flatten())
                mode_value = mode_result.mode

                if circle_id not in trackers:
                    # Create a tracker for the circle
                    tracker = choose_tracker(tracker_type)
                    tracker.init(frame, bbox)
                    trackers[circle_id] = tracker

                # Update the tracker for the circle
                success, bbox_t = trackers[circle_id].update(frame)
                if success:
                    x, y, w, h = [int(coord) for coord in bbox_t]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # Calculate center coordinates of the bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Draw the circle number in the middle of the bounding box
                    cv2.putText(frame, f'{circle_id + 1}', (center_x - 10, center_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Store circle positions for this frame
                    circle_positions.setdefault(frame_number, []).append((x + w // 2, y + h // 2))  # Center position

        # Write frame with circles to output video
        output_video.write(frame)

        # Display the frame (optional)
        #cv2_imshow(frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Write circle positions to a text file
    with open(output_txt_path, "w") as file:
        for frame_num, circles in circle_positions.items():
            file.write(f"Frame {frame_num}: ")

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Circle Tracking with CSRT/KCF trackers')

    # Add command line arguments
    parser.add_argument('--tracker_type', type=str, choices=['CSRT', 'KCF'], help='Choose tracker type')
    parser.add_argument('--input_video_path', type=str, help='Path to the input video')
    parser.add_argument('--output_video_path', type=str, help='Path to save the output video')
    parser.add_argument('--output_txt_path', type=str, help='Path to save the output circle positions text file')

    # Parse the arguments
    args = parser.parse_args()

    # Call CircleTracking with the provided arguments
    CircleTracking(args.tracker_type, args.input_video_path, args.output_video_path, args.output_txt_path)

if __name__ == "__main__":
    main()