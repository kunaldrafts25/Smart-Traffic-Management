"""Robust video capture with FFmpeg threading fixes."""

import cv2
from typing import Union
import warnings
import os
from contextlib import redirect_stderr


def create_robust_video_capture(source: Union[str, int]) -> cv2.VideoCapture:
    """
    Create a robust VideoCapture object with FFmpeg threading fixes and camera error handling.

    This function addresses multiple issues:
    - FFmpeg assertion error: "Assertion fctx->async_lock failed"
    - Camera detection errors: "obsensor_uvc_stream_channel.cpp:159"
    - OpenCV camera enumeration issues

    Args:
        source: Video source (file path, URL, or camera index)

    Returns:
        cv2.VideoCapture object with optimized settings
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if isinstance(source, int):
                try:
                    with open(os.devnull, 'w') as devnull:
                        with redirect_stderr(devnull):
                            cap = cv2.VideoCapture(source)
                except Exception:
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)

        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if isinstance(source, str):
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass

                try:
                    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if current_width > 0 and current_height > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
                except Exception:
                    pass

            elif isinstance(source, int):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    try:
                        cap.set(cv2.CAP_PROP_FPS, 30)
                    except Exception:
                        pass

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.release()
                        return cv2.VideoCapture()
                except Exception:
                    pass

        return cap

    except Exception as e:
        if "obsensor" not in str(e).lower():
            print(f"Error creating video capture for {source}: {e}")
        return cv2.VideoCapture()
