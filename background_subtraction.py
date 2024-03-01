import cv2

# Read the first frame
import cv2 as cv
from nd2reader import ND2Reader

frame1 = cv2.imread('frame1.png')

# Read the second frame
frame2 = cv2.imread('frame2.png')

# Convert frames to grayscale (if necessary)
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Perform background subtraction
subtraction_result = cv2.absdiff(frame1_gray, frame2_gray)

# Show the result of the subtraction
cv2.imshow('Subtraction Result', subtraction_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


def background_subtraction(frames, output_path="background_subtraction/"):
    first_frame = frames[0]
    background_subtracted_frames = []
    frame_index = 0
    for frame in frames:
        # Perform background subtraction by subtracting the first frame
        background_subtracted_frame = cv.absdiff(frame, first_frame)

        # Store the background-subtracted frame
        background_subtracted_frames.append(background_subtracted_frame)
        image_path = f"{output_path}frame_{frame_index:03d}.png"
        cv.imwrite(image_path, background_subtracted_frame)
        frame_index += 1

    return background_subtracted_frames


def nd2_background_subtraction(nd2_file_path, output_path="background_subtraction/"):
    first_frame = None
    background_subtracted_frames = []
    with ND2Reader(nd2_file_path) as nd2_file:
        # Print metadata
        print("Metadata:")
        print(nd2_file.metadata)
        for frame_index in range(len(nd2_file)):
            frame_data = nd2_file[frame_index]
            if first_frame is None:
                first_frame = frame_data
                # first_frame = cv.GaussianBlur(first_frame,(5,5),0)

            # Perform background subtraction by subtracting the first frame
            background_subtracted_frame = cv.absdiff(frame_data, first_frame)
            normalized_frame = cv.normalize(background_subtracted_frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            image_path = f"{output_path}frame_{frame_index:03d}.png"
            cv.imwrite(image_path, normalized_frame)
            print(f"{image_path} saved")
        return background_subtracted_frames