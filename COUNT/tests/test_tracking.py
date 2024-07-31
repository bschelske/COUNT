import tkinter as tk
import unittest

from COUNT.tracking import *
from COUNT.ui import ROISelectionApp


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
        # self.my_ui.save_overlay = tk.BooleanVar(value=True)

        self.image_h = 512

        self.nd2_file = ND2Reader_SDK(self.my_ui.file_path)
        # self.backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        self.backSub = False

    def test_DetectedObject_update(self):
        pass

    def test_DetectedObject_calculate_avg_displacement(self):
        pass

    def test_DetectedObject_enters_from_left(self):
        pass

    def test_DetectedObject_exits_right(self):
        y = 10
        x = self.my_ui.roi_x.get() + self.my_ui.roi_width.get()
        self.obj1.position = (x, y)
        result = self.obj1.exits_right(self.my_ui.roi_x.get(), self.my_ui.roi_width.get())
        self.assertEqual(result, True)

        x = self.my_ui.roi_x.get() + self.my_ui.roi_width.get() - 1
        self.obj1.position = (x, y)
        result = self.obj1.exits_right(self.my_ui.roi_x.get(), self.my_ui.roi_width.get())
        self.assertEqual(result, False)

        x = self.my_ui.roi_x.get() + self.my_ui.roi_width.get() + 1
        self.obj1.position = (x, y)
        result = self.obj1.exits_right(self.my_ui.roi_x.get(), self.my_ui.roi_width.get())
        self.assertEqual(result, True)

    def test_DetectedObject_outlet_assignment(self):
        pass

    def test_DetectedObject_center(self):
        pass

    def test_calculate_distance(self):
        result = calculate_distance(self.obj1, self.obj2)
        self.assertEqual(result, 4)

        result = calculate_distance(self.obj2, self.obj1)
        self.assertEqual(result, 4)

    def test_detect_objects(self):
        # If this fails, it could be good...
        results = []
        actuals = [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3]
        for frame in range(30):
            result, _ = detect_objects(frame_data=self.nd2_file[frame], frame_index=frame, backSub=self.backSub,
                                       ui_app=self.my_ui)
            results.append(len(result))
        self.assertEqual(results, actuals)

    def test_add_new_objects(self):
        # No previous objects
        test_dict = {}
        next_id = 1
        frame_number = 5  # arbitrary
        result_dict, result_id, = add_new_objects(new_object=self.obj1, objects_in_previous_frame_dict=test_dict,
                                                  next_new_id=next_id, frame_number=frame_number)

        intended_dict = {1: self.obj1}
        intended_id = 2
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

        # Previous objects
        test_dict = {1: self.obj1}
        next_id = 2
        frame_number = 5  # arbitrary
        result_dict, result_id, = add_new_objects(new_object=self.obj2, objects_in_previous_frame_dict=test_dict,
                                                  next_new_id=next_id, frame_number=frame_number)

        intended_dict = {1: self.obj1, 2: self.obj2}
        intended_id = 3
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

    def test_add_new_objects_edge_case(self):
        # Weird edge case?:
        # FRAME 255
        self.obj41a = DetectedObject(object_id=41, position=(451, 442), size=(9, 9), most_recent_frame=255)
        self.obj42a = DetectedObject(object_id=42, position=(444, 240), size=(9, 9), most_recent_frame=255)
        self.obj43a = DetectedObject(object_id=43, position=(266, 299), size=(9, 9), most_recent_frame=255)
        self.obj44a = DetectedObject(object_id=44, position=(0, 130), size=(2, 3), most_recent_frame=255)

        # FRAME 256
        self.obj41 = DetectedObject(object_id=41, position=(484, 442), size=(9, 9), most_recent_frame=256)
        self.obj42 = DetectedObject(object_id=42, position=(477, 240), size=(9, 9), most_recent_frame=256)
        self.obj43 = DetectedObject(object_id=43, position=(300, 299), size=(9, 9), most_recent_frame=256)
        self.obj44 = DetectedObject(object_id=44, position=(26, 127), size=(9, 9), most_recent_frame=256)

        self.obj45 = DetectedObject(object_id=45, position=(0, 188), size=(2, 3), most_recent_frame=256)
        self.objx = DetectedObject(object_id=None, position=(15, 149), size=(9, 9), most_recent_frame=256)

        dict_255 = {41: self.obj41a, 42: self.obj42a, 43: self.obj43a, 44: self.obj44a}  # FRAME 255
        dict_256 = {41: self.obj41, 42: self.obj42, 43: self.obj43, 44: self.obj44, 45: self.obj45,
                    46: self.objx}  # FRAME 256
        frame_number = 256

        next_id = 45
        frame_number = 256

        # Attempt to add missing object alone
        result_dict, result_id, = add_new_objects(new_object=self.objx, objects_in_previous_frame_dict=dict_255,
                                                  next_new_id=next_id, frame_number=frame_number)

        intended_dict = {41: self.obj41a, 42: self.obj42a, 43: self.obj43a, 44: self.obj44a, 45: self.objx}
        intended_id = 46
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

        # Attempt to add obj45 alone
        result_dict, result_id, = add_new_objects(new_object=self.obj45, objects_in_previous_frame_dict=dict_255,
                                                  next_new_id=next_id, frame_number=frame_number)

        intended_dict = {41: self.obj41a, 42: self.obj42a, 43: self.obj43a, 44: self.obj44a, 45: self.obj45}
        intended_id = 46
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

        # Iterate on dict, adding 45 then objx
        result_dict = {41: self.obj41a, 42: self.obj42a, 43: self.obj43a, 44: self.obj44a}

        result_dict, result_id, = add_new_objects(new_object=self.obj45, objects_in_previous_frame_dict=result_dict,
                                                  next_new_id=next_id, frame_number=frame_number)

        result_dict, result_id, = add_new_objects(new_object=self.objx, objects_in_previous_frame_dict=result_dict,
                                                  next_new_id=result_id, frame_number=frame_number)

        intended_dict = {41: self.obj41a, 42: self.obj42a, 43: self.obj43a, 44: self.obj44a, 45: self.obj45,
                         46: self.objx}
        intended_id = 47
        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_id, intended_id)

    def test_match_logic(self):
        # Second object is to the right
        self.assertGreater(self.obj2.position[0], self.obj1.position[0])

        # Distance between objects is less than centroid threshold
        distance = calculate_distance(self.obj2, self.obj1)
        self.assertLessEqual(distance, self.my_ui.max_centroid_distance.get())

        # Check both
        self.assertTrue(
            (self.obj2.position[0] > self.obj1.position[0]) and (distance < self.my_ui.max_centroid_distance.get()))

    def test_match_tracked_objects(self):
        # One Object
        # Match found... obj2 should match to obj1
        test_dict = {1: self.obj1}
        obj = self.obj2
        frame_number = 5
        result_bool, result_dict = match_tracked_objects(surviving_objects_dict=test_dict, object_in_frame=obj,
                                                         frame_number=frame_number, ui_app=self.my_ui)
        self.assertFalse(result_bool)

        # No match, previous object is to the right
        test_dict = {1: self.obj2}
        obj = self.obj1
        frame_number = 5
        result_bool, result_dict = match_tracked_objects(surviving_objects_dict=test_dict, object_in_frame=obj,
                                                         frame_number=frame_number, ui_app=self.my_ui)
        self.assertTrue(result_bool)

        # Object is too far away
        test_dict = {1: self.obj1}
        self.obj2.position = (self.my_ui.max_centroid_distance.get() + 1, 6)
        obj = self.obj2
        frame_number = 5
        result_bool, result_dict = match_tracked_objects(surviving_objects_dict=test_dict, object_in_frame=obj,
                                                         frame_number=frame_number, ui_app=self.my_ui)
        self.assertTrue(result_bool)

        # Object is too far away, and to the left
        test_dict = {1: self.obj1}
        self.obj1.position = (self.obj2.position[0] - self.my_ui.max_centroid_distance.get() - 1, 6)
        obj = self.obj2
        frame_number = 5
        result_bool, result_dict = match_tracked_objects(surviving_objects_dict=test_dict, object_in_frame=obj,
                                                         frame_number=frame_number, ui_app=self.my_ui)
        self.assertTrue(result_bool)

    def test_match_tracked_objects_edge_case(self):
        # Weird Edge Case (Literally)
        # FRAME 255
        self.obj41a = DetectedObject(object_id=41, position=(451, 442), size=(9, 9), most_recent_frame=255)
        self.obj42a = DetectedObject(object_id=42, position=(444, 240), size=(9, 9), most_recent_frame=255)
        self.obj43a = DetectedObject(object_id=43, position=(266, 299), size=(9, 9), most_recent_frame=255)
        self.obj44a = DetectedObject(object_id=44, position=(0, 130), size=(2, 3), most_recent_frame=255)

        # FRAME 256
        self.obj41 = DetectedObject(object_id=41, position=(484, 442), size=(9, 9), most_recent_frame=256)
        self.obj42 = DetectedObject(object_id=42, position=(477, 240), size=(9, 9), most_recent_frame=256)
        self.obj43 = DetectedObject(object_id=43, position=(300, 299), size=(9, 9), most_recent_frame=256)
        self.obj44 = DetectedObject(object_id=44, position=(26, 127), size=(9, 9), most_recent_frame=256)
        self.obj45 = DetectedObject(object_id=45, position=(0, 188), size=(2, 3), most_recent_frame=256)
        self.objx = DetectedObject(object_id=100, position=(15, 149), size=(9, 9), most_recent_frame=256)

        dict_255 = {41: self.obj41a, 42: self.obj42a, 43: self.obj43a, 44: self.obj44a}  # FRAME 255
        dict_256 = {41: self.obj41, 42: self.obj42, 43: self.obj43, 44: self.obj44, 45: self.obj45,
                    100: self.objx}  # FRAME 256
        frame_number = 256

        for obj_id, obj in dict_256.items():
            result_bool, result_dict = match_tracked_objects(surviving_objects_dict=dict_255, object_in_frame=obj,
                                                             frame_number=frame_number, ui_app=self.my_ui)
            if obj_id == 45 or obj_id == 100:
                self.assertTrue(result_bool, (obj_id, result_bool))
            else:
                self.assertFalse(result_bool, (obj_id, result_bool))

    def test_expire_objects_logic(self):
        # Expired object
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get() + 1
        self.assertTrue((frame_number - self.obj1.most_recent_frame) > self.my_ui.timeout.get())

        # Not expired object
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get()
        self.assertFalse((frame_number - self.obj1.most_recent_frame) > self.my_ui.timeout.get())

    def test_expire_objects(self):
        # Time for obj1 to expire
        self.obj1.frames_tracked = 2
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get() + 1
        test_dict = {self.obj1.object_id: self.obj1}
        test_expire_dict = {}
        result_dict, result_list = expire_objects(surviving_objects_dict=test_dict,
                                                  expired_objects_dict=test_expire_dict,
                                                  frame_number=frame_number, image_h=self.image_h, ui_app=self.my_ui)

        intended_dict = {}
        intended_expire_dict = {self.obj1.object_id: self.obj1}

        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_list, intended_expire_dict)

        # Not time for obj 1 to expire
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get()
        test_dict = {1: self.obj1}
        test_expire_dict = {}
        result_dict, result_list = expire_objects(surviving_objects_dict=test_dict,
                                                  expired_objects_dict=test_expire_dict,
                                                  frame_number=frame_number, image_h=self.image_h, ui_app=self.my_ui)

        intended_dict = {1: self.obj1}
        intended_expire_dict = {}

        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_list, intended_expire_dict)

        # Expire object that was never tracked
        self.obj1.frames_tracked = 0
        frame_number = self.obj1.most_recent_frame + self.my_ui.timeout.get() + 1
        test_dict = {1: self.obj1}
        test_expire_dict = {}
        result_dict, result_list = expire_objects(surviving_objects_dict=test_dict,
                                                  expired_objects_dict=test_expire_dict,
                                                  frame_number=frame_number, image_h=self.image_h, ui_app=self.my_ui)

        intended_dict = {}
        intended_expire_dict = {}

        self.assertEqual(result_dict, intended_dict)
        self.assertEqual(result_list, intended_expire_dict)

    def test_nd2_mog_contours(self):
        self.mog_results_dict, results_history_list = nd2_mog_contours(self.my_ui.file_path, self.my_ui)
        # print(results_dict)
        self.assertEqual(len(self.mog_results_dict), 100)

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

    def test_export_to_csv(self):
        csv_filename = 'test.csv'
        # export_to_csv(self.mog_results_dict, csv_filename)
        pass


if __name__ == '__main__':
    unittest.main()
