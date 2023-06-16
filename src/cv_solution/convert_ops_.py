import os

import cv2


def video_to_frames(video: str, path_output_dir: str) -> None:
    """
    Extracts frames from a video and saves them to a directory as 'x.png' where
    x is the frame index.

    Args:
        video (str): The file path of the input video.
        path_output_dir (str): The directory path of the output frames.

    Ex:
        video_to_frames('../somepath/myvid.mp4', '../somepath/out')
    """
    # Open the video file
    vidcap = cv2.VideoCapture(video)

    # Get the filename without the extension
    nom = os.path.splitext(os.path.basename(video))[0]

    # Initialize frame count
    count = 0

    # Loop through video and extract frames
    while vidcap.isOpened():
        success, image = vidcap.read()

        # Extract frame if it exists
        if success:
            # Write frame to output directory with index and filename
            cv2.imwrite(os.path.join(path_output_dir, f"{nom}_{count}.png"), image)

            # Increment frame count
            count += 1
        else:
            # Break the loop when no more frames are found
            break

    # Release the video and destroy all windows
    vidcap.release()
    cv2.destroyAllWindows()
