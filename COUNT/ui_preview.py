"""
UI Preview Module - Handles all preview and debugging functionality
Separated from main UI for better code organization and maintainability.
"""

import cv2
import numpy as np
from pims import ND2Reader_SDK

from COUNT import background_subtraction as bs
from COUNT import tracking


class PreviewManager:
    """Handles all preview and debugging functionality for the UI"""

    def __init__(self, ui_app):
        """Initialize with reference to the main UI app"""
        self.ui_app = ui_app
        self.preview_limit = 50
        self.cached_processed_frames = {}
        self.background_model_trained = False
        self.subtractor = None

        # Preview state
        self.preview_frame = 1
        self.cell_min_radius_preview = 0
        self.cell_max_radius_preview = 0

    def preview_edge_detection(self):
        """Main preview function - called from UI"""
        # Error handling
        if not self.ui_app.file_path and not self.ui_app.folder_path.get():
            print("Choose a file first!")
            return

        self.ui_app.input_handling()

        # Initialize preview parameters
        self.preview_frame = 1
        self.cell_min_radius_preview = self.ui_app.cell_min_radius.get()
        self.cell_max_radius_preview = self.ui_app.cell_max_radius.get()

        # Pre-process frames if background subtraction is enabled
        if self.ui_app.background_subtraction.get():
            print("Training background model and processing frames...")
            self.preprocess_all_frames()
            print("Background processing complete!")

        # Create preview window and trackbars
        cv2.namedWindow('Preview (ESC to confirm)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Preview (ESC to confirm)', 800, 800)
        cv2.createTrackbar('rad_min', 'Preview (ESC to confirm)',
                           int(self.cell_min_radius_preview), 20, self._nothing)
        cv2.createTrackbar('rad_max', 'Preview (ESC to confirm)',
                           int(self.cell_max_radius_preview), 30, self._nothing)
        cv2.createTrackbar('Frame', 'Preview (ESC to confirm)',
                           int(self.preview_frame), self.preview_limit - 1, self._nothing)

        # Add debug mode trackbar
        cv2.createTrackbar('Debug Mode', 'Preview (ESC to confirm)', 1, 1, self._nothing)

        # Main preview loop
        while True:
            # Update parameters from trackbars
            self.cell_min_radius_preview = cv2.getTrackbarPos('rad_min', 'Preview (ESC to confirm)')
            self.cell_max_radius_preview = cv2.getTrackbarPos('rad_max', 'Preview (ESC to confirm)')
            self.preview_frame = cv2.getTrackbarPos('Frame', 'Preview (ESC to confirm)')
            debug_mode = cv2.getTrackbarPos('Debug Mode', 'Preview (ESC to confirm)')

            # Ensure min doesn't exceed max
            if self.cell_min_radius_preview > self.cell_max_radius_preview:
                self.cell_min_radius_preview = self.cell_max_radius_preview
                cv2.setTrackbarPos('rad_min', 'Preview (ESC to confirm)', self.cell_max_radius_preview)

            # Generate preview frame
            if debug_mode:
                frame = self.debug_detection_pipeline()
            else:
                frame = self.preview_detection()

            cv2.imshow('Preview (ESC to confirm)', frame)

            # Update UI app parameters
            self.ui_app.cell_min_radius.set(self.cell_min_radius_preview)
            self.ui_app.cell_max_radius.set(self.cell_max_radius_preview)

            # Check for ESC key
            k = cv2.waitKey(100) & 0xFF
            if k == 27:  # ESC key
                break

        cv2.destroyAllWindows()
        self.cached_processed_frames.clear()

    def preprocess_all_frames(self):
        """Pre-process all frames once to avoid lag during preview"""
        self.subtractor = bs.BackgroundSubtractor()

        with ND2Reader_SDK(self.ui_app.files[0]) as nd2_file:
            if 'm' in nd2_file.sizes.keys():
                nd2_file.iter_axes = 'm'

            # Limit frames to prevent memory issues
            max_frames = min(self.preview_limit, len(nd2_file), 100)

            # First pass: Train background model
            print(f"Training background model with {max_frames} frames...")
            training_frames = min(50, max_frames)

            for i in range(training_frames):
                frame_data = nd2_file[i]
                self.subtractor.lightweight_running_average(frame_data)

            # Second pass: Process frames for preview
            print(f"Processing frames for preview...")
            for i in range(max_frames):
                frame_data = nd2_file[i]
                processed_frame = self.subtractor.lightweight_running_average(frame_data)
                self.cached_processed_frames[i] = processed_frame

                if i % 10 == 0:
                    print(f"Processed {i + 1}/{max_frames} frames")

            self.background_model_trained = True

    def preview_detection(self):
        """Standard preview showing final detection results"""
        with ND2Reader_SDK(self.ui_app.files[0]) as nd2_file:
            if 'm' in nd2_file.sizes.keys():
                nd2_file.iter_axes = 'm'

            # Get processed frame from cache or use original
            if (self.ui_app.background_subtraction.get() and
                    self.preview_frame in self.cached_processed_frames):
                preview_frame_processed = self.cached_processed_frames[self.preview_frame]
            else:
                preview_frame_processed = nd2_file[self.preview_frame]
                if len(preview_frame_processed.shape) == 3:
                    preview_frame_processed = cv2.cvtColor(preview_frame_processed, cv2.COLOR_BGR2GRAY)

            # Run contour detection
            contours = tracking.process_image(preview_frame_processed, self.ui_app)

            # Prepare display frame from original
            original_frame = nd2_file[self.preview_frame]
            if len(original_frame.shape) == 2:
                display_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = original_frame.copy()

            # Normalize for display
            display_frame = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Draw contours and info
            display_frame = cv2.drawContours(display_frame, contours, -1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Frame {self.preview_frame} Objects: {len(contours)}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            return display_frame

    def debug_detection_pipeline(self):
        """Debug version showing each step of the detection process"""
        with ND2Reader_SDK(self.ui_app.files[0]) as nd2_file:
            if 'm' in nd2_file.sizes.keys():
                nd2_file.iter_axes = 'm'

            original_frame = nd2_file[self.preview_frame]

            # Get processed frame
            if (self.ui_app.background_subtraction.get() and
                    self.preview_frame in self.cached_processed_frames):
                processed_frame = self.cached_processed_frames[self.preview_frame]
            else:
                processed_frame = original_frame
                if len(processed_frame.shape) == 3:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

            # Debug information
            self._print_debug_info(processed_frame)

            # Process detection steps
            canny_img = cv2.Canny(processed_frame, self.ui_app.canny_lower.get(),
                                  self.ui_app.canny_upper.get(), 3)

            contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours
            mask = np.zeros(canny_img.shape[:2], dtype=np.uint8)
            final_contours = tracking.remove_overlapped_objects(
                mask, contours, self.ui_app.cell_max_radius.get(), self.ui_app.cell_min_radius.get()
            )

            # Create debug visualization
            return self._create_debug_visualization(original_frame, processed_frame, canny_img,
                                                    contours, final_contours)

    def _print_debug_info(self, processed_frame):
        """Print debug information about the current frame"""
        print(f"Frame {self.preview_frame} stats:")
        print(f"  Shape: {processed_frame.shape}")
        print(f"  Min/Max: {processed_frame.min()}/{processed_frame.max()}")
        print(f"  Mean: {processed_frame.mean():.1f}")
        print(f"  Radius range: {self.ui_app.cell_min_radius.get()}-{self.ui_app.cell_max_radius.get()}")

    def _create_debug_visualization(self, original, processed, canny, raw_contours, final_contours):
        """Create a multi-panel debug view"""
        # Normalize all images for display
        original_display = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if len(original_display.shape) == 2:
            original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR)

        processed_display = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)

        canny_display = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        # Draw contours
        raw_contour_display = processed_display.copy()
        cv2.drawContours(raw_contour_display, raw_contours, -1, (0, 255, 0), 1)

        final_contour_display = original_display.copy()
        cv2.drawContours(final_contour_display, final_contours, -1, (0, 0, 255), 2)

        # Resize for display
        h, w = 300, 300
        panels = [original_display, processed_display, canny_display,
                  raw_contour_display, final_contour_display]
        resized_panels = [cv2.resize(panel, (w, h)) for panel in panels]

        # Add labels
        labels = [
            "1. Original",
            "2. Background Sub",
            "3. Canny Edges",
            f"4. Raw Contours ({len(raw_contours)})",
            f"5. Final ({len(final_contours)})"
        ]

        for panel, label in zip(resized_panels, labels):
            cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Create combined display (2x3 grid)
        top_row = np.hstack([resized_panels[0], resized_panels[1], resized_panels[2]])
        bottom_row = np.hstack([resized_panels[3], resized_panels[4], np.zeros_like(resized_panels[4])])

        combined = np.vstack([top_row, bottom_row])
        return combined

    def analyze_image_stats(self, frame_index=0):
        """Analyze and print detailed statistics for a specific frame"""
        with ND2Reader_SDK(self.ui_app.files[0]) as nd2_file:
            if 'm' in nd2_file.sizes.keys():
                nd2_file.iter_axes = 'm'

            frame = nd2_file[frame_index]
            print(f"\n=== Frame {frame_index} Analysis ===")
            print(f"Shape: {frame.shape}")
            print(f"Dtype: {frame.dtype}")
            print(f"Min: {np.min(frame)}")
            print(f"Max: {np.max(frame)}")
            print(f"Mean: {np.mean(frame):.1f}")
            print(f"Std: {np.std(frame):.1f}")
            print(f"1st percentile: {np.percentile(frame, 1)}")
            print(f"99th percentile: {np.percentile(frame, 99)}")
            print(f"Median: {np.median(frame):.1f}")

            # Analyze histogram
            hist, bins = np.histogram(frame.flatten(), bins=50)
            peak_bin = np.argmax(hist)
            print(f"Histogram peak at: {bins[peak_bin]:.1f}")

    def _nothing(self, x):
        """Callback for trackbars (required but does nothing)"""
        pass


# Convenience functions for integration
def create_preview_manager(ui_app):
    """Factory function to create a PreviewManager"""
    return PreviewManager(ui_app)


def add_preview_to_ui(ui_app):
    """Add preview functionality to an existing UI app"""
    ui_app.preview_manager = PreviewManager(ui_app)

    # Replace the original method with a call to the preview manager
    def preview_edge_detection():
        ui_app.preview_manager.preview_edge_detection()

    ui_app.preview_edge_detection = preview_edge_detection
    return ui_app
