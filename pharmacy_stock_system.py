import cv2
import numpy as np
import imutils
from pynput import keyboard
import cvzone
from pyzbar.pyzbar import decode
from datetime import datetime

# To detect selected medicine
highlights = {"medicineA": False, "medicineB": False, "medicineC": False}

NEAR_EXP = 10
WARN_EXP = 15
LOW_STOCK = 2
new_width = 1354
start = 375


# User Instructions
print("-------------------------Pharmacy Stock System------------------------------------------\n")
print("Click 1, 2, or 3 to view information about MedicineA, MedicineB, or MedicineC")
print("Yellow text --> Expiring within {} Days || Red text --> Expiring within {} Days".format(WARN_EXP, NEAR_EXP))
print("Yellow warning --> Medicine low in stock || Red warning --> Medicine out of stock")
print("\n----------------------------------------------------------------------------------------\n")

# Function to load [low stock] and [out of stock] warning images


def load_warning_signs():
    warning_sign = cv2.imread("src/low_stock.png", cv2.IMREAD_UNCHANGED)
    out_of_stock = cv2.imread("src/out_stock.png", cv2.IMREAD_UNCHANGED)

    warning_sign_resized = imutils.resize(
        warning_sign, width=150, inter=cv2.INTER_LINEAR)
    out_of_stock_resized = imutils.resize(
        out_of_stock, width=150, inter=cv2.INTER_LINEAR)

    return warning_sign_resized, out_of_stock_resized

# Function to overlay the frame with the warning


def display_warning_sign(frame, sign_image, position):
    hf, wf, _ = frame.shape
    hw, ww, _ = sign_image.shape
    frame = cvzone.overlayPNG(frame, sign_image, [wf - ww, hf - hw])

# Function to draw rectangles over selected (highlighted) objects


def draw_rectangles(frame, medicine_list, color, row):
    for coords in medicine_list:
        x, y, w, h = coords
        cv2.rectangle(
            frame, (x, y + hf//2*row), (x + w, y + h + hf//2*row), color, 2)

# Processes QR data and stores medicine coordinates


def process_data(info, coords):
    try:
        (name, price, date) = info.split('/')
        if len(date) == 8:
            date_format = "%d%m%Y"
            current_date = datetime.now().date()
            qr_date = datetime.strptime(date, date_format).date()
            date_difference = (qr_date - current_date).days
            date_differences[name] = date_difference
            prices[name] = price
        if name == "medicineA" and coords not in medicineA_list:
            medicineA_list.append(coords)
        elif name == "medicineB" and coords not in medicineB_list:
            medicineB_list.append(coords)
        elif name == "medicineC" and coords not in medicineC_list:
            medicineC_list.append(coords)
    except:
        pass


# QR Data + sharpness filter (sharpness to ease QR decoding during processing)
sharpness_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
date_differences = {}
prices = {}

# Coordinates - Allows drawing contours based on user command
# Lists to temporarily hold coordinates
medicineA_list = []
medicineB_list = []
medicineC_list = []

# Dictionaries to permanently hold coordinates
medicineA_coords = {'row0': [], 'row1': []}
medicineB_coords = {'row0': [], 'row1': []}
medicineC_coords = {'row0': [], 'row1': []}

# Loading of warning signs (low stock & out of stock)
warning_sign_resized, out_of_stock_resized = load_warning_signs()
hw, ww, cw = warning_sign_resized.shape
ho, wo, co = out_of_stock_resized.shape

# Default font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 0, 0)

# Load the empty shelf image and crop to correct resolution
empty_shelf = cv2.imread('src/shelf_10_full.jpg', 0)[:, start:start+new_width]
hf, wf = empty_shelf.shape
empty_shelves = [empty_shelf[
    :hf//2, :], empty_shelf[
    hf//2:, :]]

# Open a video capture object
cap = cv2.VideoCapture('src/video_full.mp4')

pressed_keys = set()
cnts = []

# Functions to detect keypresses (pynput library)


def on_press(key):
    try:
        # Convert the pressed key to a string and add it to the set
        pressed_keys.add(key.char)
    except AttributeError:
        # Handle special keys if needed
        pass


def on_release(key):
    try:
        # Remove the released key from the set
        pressed_keys.remove(key.char)
    except AttributeError:
        # Handle special keys if needed
        pass


FPSC = cvzone.FPS()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# Used to identify shelf - value changes between 0 (top row) and 1 (bottom row)
row_flag = 1
frame_skip = 0  # Used to skip every other frame
# ================================= Main Loop ================================= #
# Set up the keyboard listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:

    while True:
        frame_skip += 1
        if frame_skip % 2 == 0:
            continue
        # Read a frame from the video
        success, frame = cap.read()

        # Break the loop if the video has ended
        if not success:
            break
        frame = frame[:, start:start+new_width]

        # Change portion of image (top or bottom shelf) based on row_flag value
        if row_flag:
            frame_to_process = frame[hf//2:, :]
            empty_shelf = empty_shelves[0]
        else:
            frame_to_process = frame[:hf//2, :]
            empty_shelf = empty_shelves[1]

        # Change to gray for rest of processing
        frame_gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

        # Background Subtraction
        diff = cv2.absdiff(frame_gray, empty_shelf)
        _, thresholded = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(
            frame_to_process, frame_to_process, mask=thresholded)

        # Shadow Removal
        result_thresholded = cv2.threshold(
            result, 180, 255, cv2.THRESH_BINARY)[1]
        result_dilated = cv2.dilate(result_thresholded, None, iterations=2)

        # Frame preprocessing to find contours
        result_blur = cv2.GaussianBlur(result_dilated, (5, 5), 0)
        result_edge = cv2.Canny(result_blur, 180, 255)
        result_close = cv2.morphologyEx(
            result_edge, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(
            result_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Contour Loop
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)

            # Eliminate small and extremely wide contours
            if cv2.contourArea(cnt) < 12000 or w > frame.shape[1]//2:
                continue

            # Crop out the contour and preprocess
            temp = frame_gray[y:y+h, x:x+w]
            temp = cv2.GaussianBlur(temp, (5, 5), 0)
            temp = cv2.filter2D(temp, -1, sharpness_filter)

            # pyzbar library decoding operations
            for d in decode(temp):
                qr_data = d.data.decode('utf-8')
                process_data(qr_data, (x, y, w, h))

        # find total amount of each medicine
        medicineA_count = len(
            medicineA_coords["row0"])+len(medicineA_coords["row1"])
        medicineB_count = len(
            medicineB_coords["row0"])+len(medicineB_coords["row1"])
        medicineC_count = len(
            medicineC_coords["row0"])+len(medicineC_coords["row1"])

        # Displaying Yellow sign for low stock - Red sign for out of stock
        if medicineA_count < LOW_STOCK or medicineB_count < LOW_STOCK or medicineC_count < LOW_STOCK:
            frame = cvzone.overlayPNG(
                frame, warning_sign_resized, [wf - ww, hf - hw])

        if medicineA_count == 0 or medicineB_count == 0 or medicineC_count == 0:
            frame = cvzone.overlayPNG(
                frame, out_of_stock_resized, [wf - wo, hf - ho])

        # Check for user input to draw contours around relevant medicine
        # 1 --> medA     2 --> medB     3 --> medC
        for key, medicine in zip(['1', '2', '3'], ['medicineA', 'medicineB', 'medicineC']):
            if key in pressed_keys:
                highlights[medicine] = not highlights[medicine]

            if highlights[medicine]:
                # Try-Except statement used in case a medicine is absent -- throws error when drawing rectangles
                try:
                    # Display coordinates based on the pressed key
                    if key == '1':
                        draw_rectangles(
                            frame, medicineA_coords['row0'], (0, 255, 0), 0)
                        draw_rectangles(
                            frame, medicineA_coords['row1'], (0, 255, 0), 1)
                    elif key == '2':
                        draw_rectangles(
                            frame, medicineB_coords['row0'], (0, 255, 0), 0)
                        draw_rectangles(
                            frame, medicineB_coords['row1'], (0, 255, 0), 1)
                    elif key == '3':
                        draw_rectangles(
                            frame, medicineC_coords['row0'], (0, 255, 0), 0)
                        draw_rectangles(
                            frame, medicineC_coords['row1'], (0, 255, 0), 1)
                except AttributeError:
                    pass

        y_position = 99
        for key in prices.keys():
            label = "{}: {} OMR, exp: {} days".format(
                key, prices[key], date_differences[key]) if highlights[key] else " "
            color = (
                0, 255, 255) if date_differences[key] < WARN_EXP else font_color
            color = (0, 0, 255) if date_differences[key] < NEAR_EXP else color
            cv2.putText(frame, label, (wf-650, y_position), font,
                        font_scale, color, font_thickness)
            y_position -= 33

        count_text = {
            "medicineA": {"count": medicineA_count, "color": font_color},
            "medicineB": {"count": medicineB_count, "color": font_color},
            "medicineC": {"count": medicineC_count, "color": font_color}
        }

        # Display count
        y_position = 240
        for key, value in count_text.items():
            line = f"{key}: {value['count']} in stock"
            value['color'] = (
                (0, 255, 0)) if highlights[key] else font_color
            cv2.putText(frame, line, (wf-350, y_position), font,
                        font_scale, value['color'], font_thickness)
            y_position += 30

        # Display Frame
        frame = FPSC.update(frame, pos=(20, 50),
                            color=font_color, thickness=font_thickness)[1]
        cv2.imshow('frame', imutils.resize(frame, width=700))

        # Update coordinates with temporary list
        medicineA_coords["row{}".format(row_flag)] = medicineA_list
        medicineB_coords["row{}".format(row_flag)] = medicineB_list
        medicineC_coords["row{}".format(row_flag)] = medicineC_list

        # Flush Coordinates from temporary list
        medicineA_list = []
        medicineB_list = []
        medicineC_list = []

        # Invert row_flag (access other row next frame)
        row_flag = row_flag ^ 1

        # Exit if 'q' is pressed
        key = cv2.waitKey(30)
        if key == ord('q'):
            break


# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
