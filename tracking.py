import cv2 as cv

from main import detect_objects


def tracking(frames, output_path, ROI, spots, save_overlay=False):
    font = cv.FONT_HERSHEY_SIMPLEX
    overlayed_frames = []
    roi_x, roi_y, roi_h, roi_w = ROI
    top = []
    bottom = []
    active_ids = {}
    threshold_distance = 50
    img_h, img_w = frames[0].shape
    timeout_threshold = 7 * 1  # Frames
    next_id = 1

    for frame_index, frame in enumerate(frames):
        objects, img_copy = detect_objects(frame, frame_index, ROI, spots)

        # Remove IDs of objects that have moved off the screen
        for obj_id, tracked_obj in list(active_ids.items()):
            if tracked_obj.position[0] > img_w:
                del active_ids[obj_id]
            elif frame_index - tracked_obj.most_recent_frame > timeout_threshold:
                del active_ids[obj_id]  # Expire IDs if no new position found

        for obj in objects:
            match_found = False
            for obj_id, tracked_obj in active_ids.items():
                distance = calculate_distance(obj, tracked_obj)
                if distance < threshold_distance and obj.position[0] > tracked_obj.position[0]:
                    tracked_obj.id = obj.id
                    tracked_obj.update_position(obj.position)
                    tracked_obj.most_recent_frame = frame_index  # Update last frame detected
                    match_found = True
                    break

            if not match_found and obj.id is None:
                if obj.enters_from_left(roi_x):
                    obj.id = next_id
                    obj.most_recent_frame = frame_index  # Set last frame detected
                    active_ids[next_id] = obj
                    next_id += 1

        # Draw IDs
        for obj_id, tracked_obj in active_ids.items():
            cv.putText(img_copy, str(obj_id), tracked_obj.position, font, 1, (255, 255, 255), 1, cv.LINE_AA)

        # Draw ROI on every frame
        # cv.rectangle(img_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255,0,0), 2)
        overlayed_frames.append(img_copy)

    print(f'Total Cells: {len(active_ids)}\nTop: {len(top)}\nBottom: {len(bottom)}')

    if save_overlay:
        for idx, overlay_frame in enumerate(overlayed_frames):
            save_path = output_path + f"{idx}.png"
            cv.imwrite(save_path, overlay_frame)
        return None
    else:
        return overlayed_frames


def spot_correction(input_edges, spots):
    for spot in spots:
        x, y, w, h = spot
        input_edges[y:y + h, x:x + w] = 0
    return input_edges


class DetectedObject:
    def __init__(self, id, position, size, most_recent_frame):
        self.id = id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame


    def update_position(self, new_position):
        self.position = new_position

    def enters_from_left(self, roi_x):
        if roi_x < self.position[0] < roi_x + 60:
            return True
        else:
            return False

    def exits_right(self, roi_x, roi_w):
        if roi_x + roi_w < self.position[0]:
            return True
        else:
            return False

    def color(self, roi_h, roi_y):
        if int(self.position[1]) > (roi_h / 2 + roi_y):
            return (0, 150, 255)  # orange
        if int(self.position[1]) < (roi_h / 2 + roi_y):
            return (255, 150, 0)  # blue

    def center(self):
        return self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2


def calculate_distance(detectedobject1, detectedobject2):
    possible_center = detectedobject1.center()
    tracked_center = detectedobject2.center()
    distance = int(
        (((possible_center[0] - tracked_center[0]) ** 2) + ((possible_center[1] - tracked_center[1]) ** 2)) ** 0.5)
    return distance