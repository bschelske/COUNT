import cv2 as cv
import numpy as np
from nd2reader import ND2Reader
import tempfile
import shutil


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
        print(nd2_file)
        for frame_index in range(len(nd2_file)):
            frame_data = nd2_file[frame_index]
            if first_frame is None:
                first_frame = frame_data
                # first_frame = cv.GaussianBlur(first_frame,(65,65),0) # Doesn't seem to help

            # Perform background subtraction by subtracting the first frame
            background_subtracted_frame = cv.absdiff(frame_data, first_frame)
            normalized_frame = cv.normalize(background_subtracted_frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            image_path = f"{output_path}/frame_{frame_index:03d}.png"
            cv.imwrite(image_path, normalized_frame)
            print(f"{image_path} saved")
        return background_subtracted_frames


def nd2_background_subtraction_comparison(nd2_file_path, gaussian_size, output_path="background_subtraction/"):
    first_frame = None
    background_subtracted_frames = []
    with ND2Reader(nd2_file_path) as nd2_file:
        # Print metadata
        print("Metadata:")
        print(nd2_file.metadata)
        for frame_index in range(1):
            frame_data = nd2_file[frame_index]
            if first_frame is None:
                first_frame = frame_data
                first_frame = cv.GaussianBlur(first_frame,gaussian_size,0)

            # Perform background subtraction by subtracting the first frame
            background_subtracted_frame = cv.absdiff(frame_data, first_frame)
            normalized_frame = cv.normalize(background_subtracted_frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

            # Resize the image to fit the display window
            scale_percent = 40  # Adjust this value as needed (e.g., 50 for 50% reduction)
            width = int(normalized_frame.shape[1] * scale_percent / 100)
            height = int(normalized_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv.resize(normalized_frame, dim, interpolation=cv.INTER_AREA)

            cv.imshow('Subtraction Result', resized_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return background_subtracted_frames
#
#
# nd2_file_path = "nd2_files/23Feb2024 Non RosetteSep 5kHz.nd2"
# nd2_background_subtraction_comparison(nd2_file_path, (65,65))
