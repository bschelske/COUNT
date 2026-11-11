"""
Reading video files (.nd2 and .avi) as greyscale frames.

    with open_video(path) as frames:
        len(frames)                                # number of frames
        frames[i]                                  # frame i as a greyscale image
        read_region(frames, i, (x0, y0, x1, y1))   # just part of frame i (faster for uncompressed AVI files)

Uncompressed AVI files (as saved by NIS-Elements) are read directly from the file: any frame can be read
without reading the ones before it, and only the rows that are needed are read. Other AVI files (compressed
ones, such as MJPG) are read with OpenCV, and .nd2 files with the Nikon SDK (pims_nd2).
"""
import contextlib
import os
import struct
import typing

import cv2 as cv
import numpy as np

VIDEO_EXTENSIONS = (".nd2", ".avi")  # file types COUNT can track

Region = typing.Tuple[int, int, int, int]  # (x0, y0, x1, y1): columns x0..x1-1 and rows y0..y1-1


def is_supported_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


class NotRawAvi(Exception):
    """The AVI file is not an uncompressed video that RawAviReader can read."""


class RawAviReader:
    """
    Frames of an uncompressed (8-, 24- or 32-bit RGB) AVI file, read straight from the file.

    The file is scanned once for the position of every stored frame, including OpenDML (larger than 1 GB)
    files. Index entries with no data (which repeat the previous image during playback) are not frames.
    Frames are converted to greyscale exactly as OpenCV does, so the result matches AviReader.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "rb")
        try:
            self._parse()
        except Exception:
            self._file.close()
            raise

    # ---- parsing ----
    def _parse(self):
        file_size = os.fstat(self._file.fileno()).st_size
        self._stream = None       # index of the first video stream
        self._format = None       # BITMAPINFOHEADER fields
        self._palette = None
        self.offsets = []         # file offset of each frame's data
        streams_seen = 0

        def walk(start, end, in_movi):
            nonlocal streams_seen
            pos = start
            while pos + 8 <= end:
                self._file.seek(pos)
                header = self._file.read(12)
                if len(header) < 8:
                    break
                chunk_id, size = header[:4], struct.unpack("<I", header[4:8])[0]
                if chunk_id in (b"RIFF", b"LIST"):
                    list_type = header[8:12]
                    if list_type in (b"AVI ", b"AVIX", b"hdrl", b"movi", b"rec "):
                        walk(pos + 12, min(pos + 8 + size, end), in_movi or list_type == b"movi")
                    elif list_type == b"strl":
                        self._read_stream_header(pos + 12, min(pos + 8 + size, end), streams_seen)
                        streams_seen += 1
                elif in_movi and size > 0 and chunk_id[2:] in (b"db", b"dc"):
                    if self._stream is not None and chunk_id[:2] == b"%02d" % self._stream:
                        if size < self._frame_bytes:
                            raise NotRawAvi("frames are smaller than an uncompressed image (compressed video)")
                        self.offsets.append(pos + 8)
                pos += 8 + size + (size & 1)

        walk(0, file_size, False)
        if self._format is None:
            raise NotRawAvi("no uncompressed video stream")
        if not self.offsets:
            raise NotRawAvi("no frames")

    def _read_stream_header(self, start, end, stream_number):
        """Read one stream's strh/strf. Keeps the first video stream if it is uncompressed RGB."""
        if self._stream is not None:
            return
        self._file.seek(start)
        data = self._file.read(end - start)
        strh = data.find(b"strh")
        if strh < 0 or data[strh + 8:strh + 12] != b"vids":
            return
        strf = data.find(b"strf")
        if strf < 0:
            raise NotRawAvi("video stream has no format")
        strf_size = struct.unpack("<I", data[strf + 4:strf + 8])[0]
        header = data[strf + 8:strf + 8 + strf_size]
        (bi_size, width, height, planes, bits, compression, size_image,
         _, _, colours_used, _) = struct.unpack("<IiiHHIIiiII", header[:40])
        if compression not in (0, 3) or bits not in (8, 24, 32) or width <= 0 or height == 0:
            raise NotRawAvi("video is compressed or not 8/24/32-bit")
        if compression == 3 and bits != 32:
            raise NotRawAvi("unsupported bitfield format")
        self._stream = stream_number
        self.width, self.height = width, abs(height)
        self._bottom_up = height > 0
        self._channels = bits // 8
        self._stride = ((width * bits + 31) // 32) * 4
        self._frame_bytes = self._stride * self.height
        self._format = (width, height, bits)
        if bits == 8:
            n_colours = colours_used or 256
            palette = np.frombuffer(header[bi_size:bi_size + 4 * n_colours], np.uint8)
            if len(palette) < 4 * n_colours:
                raise NotRawAvi("8-bit video without a palette")
            bgr = palette.reshape(-1, 4)[:, :3].copy()
            lookup = np.zeros(256, np.uint8)
            lookup[:len(bgr)] = cv.cvtColor(bgr.reshape(1, -1, 3), cv.COLOR_BGR2GRAY).ravel()
            self._palette = lookup

    # ---- reading ----
    @property
    def shape(self):
        return self.height, self.width

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index: int):
        return self.read_region(index, (0, 0, self.width, self.height))

    def read_region(self, index: int, region: Region):
        """Rows y0..y1-1 and columns x0..x1-1 of frame `index`, as a greyscale image."""
        if index < 0:
            index += len(self.offsets)
        if not 0 <= index < len(self.offsets):
            raise IndexError(f"frame {index} is outside {self.path} ({len(self.offsets)} frames)")
        x0, y0, x1, y1 = region
        first_row = (self.height - y1) if self._bottom_up else y0  # first stored row needed
        n_rows = y1 - y0
        self._file.seek(self.offsets[index] + first_row * self._stride)
        data = self._file.read(n_rows * self._stride)
        if len(data) < n_rows * self._stride:
            raise IOError(f"frame {index} of {self.path} is incomplete")
        rows = np.frombuffer(data, np.uint8).reshape(n_rows, self._stride)
        if self._bottom_up:
            rows = rows[::-1]
        pixels = rows[:, x0 * self._channels:x1 * self._channels]
        if self._channels == 1:
            return self._palette[pixels]
        pixels = np.ascontiguousarray(pixels).reshape(n_rows, x1 - x0, self._channels)
        code = cv.COLOR_BGR2GRAY if self._channels == 3 else cv.COLOR_BGRA2GRAY
        return cv.cvtColor(pixels, code)

    def close(self):
        self._file.close()


class AviReader:
    """
    The frames of an AVI file as greyscale images, read with OpenCV (used for compressed AVI files).

    Frames are decoded in order, which is how COUNT reads them; asking for an earlier frame reopens the file.
    """

    def __init__(self, path: str):
        self.path = path
        capture = cv.VideoCapture(path)
        if not capture.isOpened():
            raise IOError(f"Could not open video file {path}")
        # Count frames by skipping through the file: the frame count stored in an AVI header can be wrong
        self._length = 0
        while capture.grab():
            self._length += 1
        capture.release()
        self._capture = None
        self._next_frame = 0
        self._open()

    def _open(self):
        if self._capture is not None:
            self._capture.release()
        self._capture = cv.VideoCapture(self.path)
        self._next_frame = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index: int):
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"frame {index} is outside {self.path} ({self._length} frames)")
        if index < self._next_frame:
            self._open()
        while self._next_frame < index:  # skip ahead without decoding
            self._capture.grab()
            self._next_frame += 1
        ok, frame = self._capture.read()
        if not ok:
            raise IOError(f"Could not read frame {index} of {self.path}")
        self._next_frame += 1
        if frame.ndim == 3:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return frame

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None


@contextlib.contextmanager
def open_video(path: str):
    """
    Open an .nd2 or .avi file. Use as `with open_video(path) as frames:`; `frames[i]` is frame i as a
    greyscale image and `len(frames)` is the number of frames.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == ".avi":
        try:
            reader = RawAviReader(path)
        except (NotRawAvi, struct.error, ValueError):
            reader = AviReader(path)
        try:
            yield reader
        finally:
            reader.close()
    elif extension == ".nd2":
        from pims import ND2Reader_SDK  # imported here so AVI files and the tests work without the Nikon SDK
        with ND2Reader_SDK(path) as reader:
            if 'm' in reader.sizes.keys():  # new nikon weirdness
                reader.iter_axes = 'm'
            yield reader
    else:
        raise ValueError(f"Unsupported file type {extension!r} ({path}). COUNT reads .nd2 and .avi files.")


def frame_shape(frames) -> typing.Tuple[int, int]:
    """(height, width) of the frames in an open video."""
    shape = getattr(frames, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        return shape
    return tuple(np.asarray(frames[0]).shape[:2])


def read_region(frames, index: int, region: typing.Optional[Region] = None):
    """Frame `index` of an open video, cropped to `region` (x0, y0, x1, y1), or the whole frame if None."""
    if region is None:
        return np.asarray(frames[index])
    if hasattr(frames, "read_region"):
        return frames.read_region(index, region)
    x0, y0, x1, y1 = region
    return np.asarray(frames[index])[y0:y1, x0:x1]
