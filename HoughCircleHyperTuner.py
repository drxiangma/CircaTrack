# Circle Detection with HoughCircles
# Fine-tune parameters dp, minDist, param1, param2, Gaussian blur kernel size
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Function to calculate loss
def loss_fn(actual_count, target_count):
    se = (actual_count - target_count) ** 2
    if actual_count > target_count:
        excess = actual_count - target_count
        penalty = excess ** 2
        se = se + penalty * 2
    return se

def finetune_hp(input_video, output_video):
    # Parameters to fine-tune
    dp_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    min_dist_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    param1_values = [10, 20, 30, 40, 50, 60, 70, 80]
    param2_values = [10, 20, 30, 40, 50, 60, 70, 80]
    gaussian_blur_kernel = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]

    # Initialize variables for best parameters and minimum loss
    best_parameters = {}
    min_loss = float('inf')
    
    # 10 circles as target
    target_count = 10

    for dp in dp_values:
        for min_dist in min_dist_values:
            for param1 in param1_values:
                for param2 in param2_values:
                    for kernel_size in gaussian_blur_kernel:
                        frame_number = 0
                        video = cv2.VideoCapture(input_video)
                        loss = 0
                        higher_count = 0
                        # Arrays to store detected circle counts
                        detected_counts = []

                        while True:
                            ret, frame = video.read()
                            if not ret:
                                break

                            frame_number += 1
                            
                            # Convert frame to grayscale for circle detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            # Apply GaussianBlur to reduce noise
                            gray_blurred = cv2.GaussianBlur(gray, kernel_size, 0)

                            # Detect circles using HoughCircles
                            circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                                                        param1=param1, param2=param2, minRadius=20, maxRadius=150)

                            if circles is not None:
                                circles = np.round(circles[0, :]).astype("int")
                                circle_count = len(circles)
                                detected_counts.append(circle_count)
                                loss += loss_fn(circle_count, target_count)
                                #print(f'Number of detected circles: {len(circles)} loss: {loss}')
                                if circle_count > target_count:
                                    higher_count += 1
                                # Early stopping if loss is too high
                                if loss > 600:
                                    break

                        if frame_number < 153 or loss > 600:
                            break

                        #print(f'dp {dp}, min_dist {min_dist}, param1 {param1}, param2 {param2}, kernel_size {kernel_size}, loss: {loss}, larger than target: {higher_count}')
                        # Update best parameters if loss improves
                        if loss < min_loss:
                            min_loss = loss
                            best_parameters = {'dp': dp, 'min_dist': min_dist, 'param1': param1, 'param2': param2,
                                                'kernel_size': kernel_size}
                            if loss < 600:
                                plt.hist(detected_counts, facecolor='blue', alpha=0.65)
                                plt.xlabel('Detected Circle Counts')
                                plt.ylabel('Frequency')
                                plt.xlim(0, 20)
                                plt.xticks(range(0, 20))
                                plt.title('Histogram of Detected Circle Counts')
                                plt.show()
                                plt.clf()

    # Use best parameters to detect circles and draw them
    # Open the video file again to start from the beginning
    video = cv2.VideoCapture(input_video)
    
    # Output video settings
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, best_parameters['kernel_size'], 0)

        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=best_parameters['dp'],
                                   minDist=best_parameters['min_dist'], param1=best_parameters['param1'],
                                   param2=best_parameters['param2'], minRadius=20, maxRadius=150)

        # Draw circles if detected
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for i, (x, y, r) in enumerate(circles):
                cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)
                cv2.putText(frame, f'{i+1}', (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write frame with circles to output video
        output_video.write(frame)

    # Release resources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    finetune_hp("Dot_Track_Vid_2023.mp4", "Circle_Detection.mp4")
