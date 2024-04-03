def tracking(frames, output_path, ROI, spots, save_overlay=False):
    font = cv.FONT_HERSHEY_SIMPLEX
    overlayed_frames = []
    tracked_objects = {}
    roi_x, roi_y, roi_h, roi_w = ROI
    distance_threshold = 100
    canny_lower = 200
    canny_upper = 600
    retired_ids = []
    top = []
    bottom = []
    img_h, img_w = frames[0].shape
    timeout_threshold = 5
    next_id = 1
    active_ids = {}

    for frame_index, frame in enumerate(frames):
        objects, img_copy = detect_objects(frame, ROI, canny_lower, canny_upper, spots)

        for obj in objects:
            if obj.enters_from_left():
                obj.object_id = next_id
                active_ids[next_id] = {"object": obj, "last_detected": frame_index}
                next_id += 1

            # determine_object_color(y, roi_y)

            # Update tracking and object positions
            for obj_id, bbox in tracked_objects.items():
                if obj_id not in retired_ids:
                    prev_x, prev_y, prev_w, prev_h = bbox[-1]  # Get last known bounding box
                    new_center = (x + w // 2, y + h // 2)
                    prev_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)
                    distance = int(
                        (((new_center[0] - prev_center[0]) ** 2) + ((new_center[1] - prev_center[1]) ** 2)) ** 0.5)

                    # MATCH! retain ID
                    if distance < distance_threshold and (x > prev_x):
                        tracked_objects[obj_id].append((x, y, w, h))
                        if x > (img_w - 60):  # Retire if near edge of image
                            retired_ids.append(obj_id)
                            if y > img_h // 2:
                                bottom.append(obj_id)
                            else:
                                top.append(obj_id)
                        # cv.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                        cv.putText(img_copy, str(obj_id), (x, y), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                        found_match = True
                        break

            # Check for lost objects
            current_frame = frame_index
            lost_ids = []
            for obj_id, obj_info in active_ids.items():
                last_detected = obj_info["last_detected"]
                if current_frame - last_detected > timeout_threshold:
                    lost_ids.append(obj_id)

            # Expire lost IDs
            for lost_id in lost_ids:
                del active_ids[lost_id]

            # Check if object is already tracked
            found_match = False

            # Calculate centers loop, check distances for ID

            # # If no match found, assign a new ID
            # if not found_match and (roi_x < obj.x < roi_x + 60):
            #     new_id = max(tracked_objects.keys(), default=0) + 1
            #     tracked_objects[new_id] = [(x, y, w, h)]
            #     # cv.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            #     cv.putText(img_copy, str(new_id), (x, y), font, 1, (255, 255, 255), 1, cv.LINE_AA)
            # else:
            #     pass
        # Draw ROI
        # cv.rectangle(img_copy, (roi_x, roi_y), ((roi_x+40), (roi_y+roi_h)), (0, 0, 255), 1) # Assignment region
        # # cv.rectangle(img_copy, (roi_x+40, 0), ((img_w), (img_h)), (0, 0, 255), 1) # Tracking region
        # cv.rectangle(img_copy, (img_w - 60, 0), ((60), (img_h)), (0, 0, 255), 1) # retirement region
        overlayed_frames.append(img_copy)
    print(
        f'Total Cells: {len(tracked_objects)}\nTotal Retired: {len(retired_ids)}\n{retired_ids}\nTop: {len(top)}\nBottom: {len(bottom)}')
    if save_overlay:
        for idx, overlay_frame in enumerate(overlayed_frames):
            save_path = output_path + f"{idx}.png"
            cv.imwrite(save_path, overlay_frame)
        return None
    else:
        return overlayed_frames
