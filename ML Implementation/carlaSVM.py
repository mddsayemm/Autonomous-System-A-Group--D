#!/usr/bin/env python3
"""
CARLA template: constant forward speed + steering from SVM model.

This version uses a trained SVR model from groupAsvm.joblib.
"""

import carla
import random
import time
import sys
import math
import numpy as np
import cv2
import joblib
import os

# ------------------------ CONFIGURATION --------------------------------------
HOST            = "localhost"
PORT            = 2000
TIMEOUT_S       = 5.0
THROTTLE        = 0.4
DEFAULT_STEER   = 0.0
PRINT_EVERY_N   = 30
MODEL_PATH      = r"C:\Users\mcsmu\Desktop\Line Following SVM\groupAsvm.joblib"
# -----------------------------------------------------------------------------


# ------------------------ HSV Range for Green Detection ----------------------
LOWER_HSV = np.array([70, 50, 50])
UPPER_HSV = np.array([90, 255, 255])
# -----------------------------------------------------------------------------


# ------------------------ Load the Trained Model -----------------------------
model = joblib.load(MODEL_PATH)
# -----------------------------------------------------------------------------


def extract_features(image):
    """Extract 3-band normalized x-position features from the green line."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    height, width = mask.shape

    bands = [mask[int(height * 0.6):int(height * 0.7), :],
             mask[int(height * 0.7):int(height * 0.8), :],
             mask[int(height * 0.8):, :]]

    features = []
    for band in bands:
        indices = np.column_stack(np.where(band > 0))
        if indices.size > 0:
            avg_x = np.mean(indices[:, 1]) / width * 2 - 1  # Normalize to [-1, 1]
        else:
            avg_x = 0.0
        features.append(avg_x)
    return features


def predict_steering(img):
    """
    Predict steering using the trained SVM model.
    CARLA image -> OpenCV BGR -> extract 3 features -> model.predict()
    """
    img_array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    bgr = img_array[:, :, :3]
    features = extract_features(bgr)
    steering = float(model.predict([features])[0])
    return max(-1.0, min(1.0, steering))


def parent_of(actor):
    if hasattr(actor, "get_parent"):
        return actor.get_parent()
    return getattr(actor, "parent", None)

def ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def pick_center_camera(world, vehicle):
    v_yaw = vehicle.get_transform().rotation.yaw
    best = None
    for s in world.get_actors().filter("sensor.camera.rgb"):
        p = parent_of(s)
        if p and p.id == vehicle.id:
            delta = abs(ang_diff_deg(s.get_transform().rotation.yaw, v_yaw))
            if best is None or delta < best[0]:
                best = (delta, s)
    return best[1] if best else None


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world = client.get_world()

    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        print("No vehicles found. Start a scene first.")
        return
    vehicle = vehicles[0]
    print("Controlling vehicle id=%d type=%s" % (vehicle.id, vehicle.type_id))
    vehicle.set_autopilot(False)

    camera = pick_center_camera(world, vehicle)
    if camera is None:
        print("No center RGB camera attached to the vehicle.")
        return
    print("Using camera id=%d for live feed" % camera.id)

    state = {"frames": 0, "first_ts": None, "latest_img": None}

    def cam_cb(img):
        state["latest_img"] = img
        state["frames"] += 1
        if state["frames"] % PRINT_EVERY_N == 0:
            if state["first_ts"] is None:
                state["first_ts"] = img.timestamp
            elapsed = img.timestamp - state["first_ts"]
            fps = state["frames"] / elapsed if elapsed else 0.0
            print("camera frames: %d   %.1f FPS" % (state["frames"], fps))

    camera.listen(cam_cb)

    try:
        while True:
            img = state["latest_img"]
            if img is not None:
                steer = predict_steering(img)
            else:
                steer = DEFAULT_STEER
            vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE, steer=steer))
            time.sleep(0.01)  # 100 Hz loop

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        camera.stop()
        vehicle.apply_control(carla.VehicleControl(brake=1.0))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.stderr.write("[ERROR] " + str(err) + "\n"
                         "Is the CARLA server running on this host/port?\n")
