import unittest
import tkinter as tk
from COUNT.tracking import *
from COUNT.ui import ROISelectionApp

# TODO: Add all functions from COUNT.tracking
# TODO: Expand upon test cases for each function

class TestTracking(unittest.TestCase):
    def setUp(self):
        self.obj1 = DetectedObject(object_id=1, position=(1, 2), size=(2, 2), most_recent_frame=1)
        self.obj2 = DetectedObject(object_id=2, position=(3, 6), size=(2, 2), most_recent_frame=1)

        self.my_ui = ROISelectionApp(tk.Tk())
        self.my_ui.file_path = r'C:\Users\bensc\PycharmProjects\COUNT_synth\100_avi.nd2'
        self.my_ui.cell_radius = tk.IntVar(value=3)
        self.my_ui.roi_height = tk.IntVar(value=512)
        self.my_ui.roi_width = tk.IntVar(value=512)
        self.my_ui.roi_x = tk.IntVar(value=0)

        self.image_h = 512

        self.nd2_file = ND2Reader_SDK(self.my_ui.file_path)
        # self.backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        self.backSub = False

    def test_calculate_distance(self):
        result = calculate_distance(self.obj1, self.obj2)
        self.assertEqual(result, 4)

        result = calculate_distance(self.obj2, self.obj1)
        self.assertEqual(result, 4)

    def test_detect_objects(self):
        # If this fails, it could be good...
        results = []
        actuals = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3]
        for frame in range(30):
            result, _ = detect_objects(frame_data=self.nd2_file[frame], frame_index=frame, backSub=self.backSub,
                                       ui_app=self.my_ui)
            results.append(len(result))
        self.assertEqual(results, actuals)

    def test_add_new_objects(self):
        # No previous objects
        test_dict = {}
        next_id = 2
        frame_number = 5  # arbitrary
        result_dict, result_id, = add_new_objects(object_in_frame=self.obj1, objects_in_previous_frame_dict=test_dict, next_new_id=next_id, frame_number=frame_number)

        intended_dict = {2: self.obj1}
        intended_id = next_id + 1
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

        # Previous objects
        test_dict = {1: self.obj1}
        next_id = 2
        frame_number = 5  # arbitrary
        result_dict, result_id, = add_new_objects(object_in_frame=self.obj2, objects_in_previous_frame_dict=test_dict, next_new_id=next_id, frame_number=frame_number)

        intended_dict = {1: self.obj1, 2: self.obj2}
        intended_id = next_id + 1
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

    def test_match_logic(self):
        # Second object is to the right
        self.assertGreater(self.obj2.position[0], self.obj1.position[0])

        # Distance between objects is less than centroid threshold
        distance = calculate_distance(self.obj2, self.obj1)
        self.assertLessEqual(distance, self.my_ui.max_centroid_distance.get())

        # Check both
        self.assertTrue((self.obj2.position[0] > self.obj1.position[0]) and (distance < self.my_ui.max_centroid_distance.get()))

    def test_match_tracked_objects(self):
        # One Object
        # Match found... obj2 should match to obj1
        test_dict = {1: self.obj1}
        obj = self.obj2
        frame_number = 5
        result_bool = match_tracked_objects(objects_in_previous_frame_dict=test_dict, object_in_frame=obj, frame_number=frame_number, ui_app=self.my_ui)
        self.assertFalse(result_bool)

        # No match, previous object is to the right
        test_dict = {1: self.obj2}
        obj = self.obj1
        frame_number = 5
        result_bool = match_tracked_objects(objects_in_previous_frame_dict=test_dict, object_in_frame=obj, frame_number=frame_number, ui_app=self.my_ui)
        self.assertTrue(result_bool)

        # Object is too far away
        test_dict = {1: self.obj1}
        self.obj2.position = (self.my_ui.max_centroid_distance.get() + 1, 6)
        obj = self.obj2
        frame_number = 5
        result_bool = match_tracked_objects(objects_in_previous_frame_dict=test_dict, object_in_frame=obj, frame_number=frame_number, ui_app=self.my_ui)
        self.assertTrue(result_bool)

        # Object is too far away, and to the left
        test_dict = {1: self.obj1}
        self.obj1.position = (self.obj2.position[0] - self.my_ui.max_centroid_distance.get() - 1, 6)
        obj = self.obj2
        frame_number = 5
        result_bool = match_tracked_objects(objects_in_previous_frame_dict=test_dict, object_in_frame=obj, frame_number=frame_number, ui_app=self.my_ui)
        self.assertTrue(result_bool)

    def test_expire_objects_logic(self):
        # Expired object
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get() + 1
        self.assertTrue((frame_number - self.obj1.most_recent_frame) > self.my_ui.timeout.get())

        # Not expired object
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get()
        self.assertFalse((frame_number - self.obj1.most_recent_frame) > self.my_ui.timeout.get())

    def test_expire_objects(self):
        # Time for obj1 to expire
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get() + 1
        test_dict = {1: self.obj1}
        test_list = []
        result_dict, result_list = expire_objects(objects_in_frame_dict=test_dict, object_final_position=test_list, frame_number=frame_number, image_h=self.image_h, ui_app=self.my_ui)

        intended_dict = {}
        intended_list = [self.obj1]

        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_list, intended_list)

        # Not time for obj 1 to expire
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get()
        test_dict = {1: self.obj1}
        test_list = []
        result_dict, result_list = expire_objects(objects_in_frame_dict=test_dict, object_final_position=test_list,
                                                  frame_number=frame_number, image_h=self.image_h, ui_app=self.my_ui)

        intended_dict = {1: self.obj1}
        intended_list = []

        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_list, intended_list)

        # TODO: There must be some kind of edge case here

    def test_nd2_mog_contours(self):
        results_list, results_history_list = nd2_mog_contours(self.my_ui.file_path, self.my_ui)
        self.assertEqual(len(results_list), 100)

    def test_coord_in_roi(self):
        # self.my_ui ROI: 0,0 512, 512
        coord = (0, 0)
        self.assertEqual(coord_in_roi(coord, self.my_ui), True)

        coord = (10, 0)
        self.assertEqual(coord_in_roi(coord, self.my_ui), True)

        coord = (0, 10)
        self.assertEqual(coord_in_roi(coord, self.my_ui), True)

        coord = (-10, 0)
        self.assertEqual(coord_in_roi(coord, self.my_ui), False)

        coord = (0, -10)
        self.assertEqual(coord_in_roi(coord, self.my_ui), False)

        coord = (-10, -10)
        self.assertEqual(coord_in_roi(coord, self.my_ui), False)

    def test_ui_works(self):
        print(self.my_ui.canny_upper.get())


if __name__ == '__main__':
    unittest.main()