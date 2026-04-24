from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re

card_model   = YOLO('Models/card_detector.pt')
nid_model     = YOLO('Models/nid_detector.pt')
digit_model = YOLO('Models/digit_detector.pt')
reader       = easyocr.Reader(['ar'], gpu=False)
clahe       = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) 


#detect and crop the card from the input image
def crop_card(image):
    prediction_results = card_model.predict(image, conf=0.5, iou=0.45)
    prediction_result  = prediction_results[0]

    if len(prediction_result.boxes) == 0:
        return None

    boxes = prediction_result.boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

    scores = confs * areas
    best_idx = np.argmax(scores)

    h, w = image.shape[:2]

    x1, y1, x2, y2 = boxes_xyxy[best_idx].astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    image_cropped = image[y1:y2, x1:x2].copy()
    
    return image_cropped


#correct the orientation of the cropped card
def correct_orientation(image_cropped):
    angles = [0, 90, 180, 270]
    best_angle = 0
    max_conf = 0

    h, w = image_cropped.shape[:2]
    small = cv2.resize(image_cropped, (320, 320))

    for angle in angles:
        if angle == 90:
            rotated = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(small, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = small

        results = nid_model.predict(rotated, conf=0.4, verbose=False)[0]

        if len(results.boxes) > 0:
            current_conf = results.boxes.conf.cpu().numpy().max()

            if current_conf > max_conf:
                max_conf = current_conf
                best_angle = angle

    if best_angle == 90:
        return cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
    elif best_angle == 180:
        return cv2.rotate(image_cropped, cv2.ROTATE_180)
    elif best_angle == 270:
        return cv2.rotate(image_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image_cropped


def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)

    if lines is None:
        return image

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90

        if -20 < angle < 20:
            angles.append(angle)

    if len(angles) < 5:  
        return image

    median_angle = np.median(angles)

    if abs(median_angle) < 2 or abs(median_angle) > 20:
        return image  

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    return rotated
    
    
#detect and crop the ID region from the corrected image
def crop_id_box(image, pad=5):
    result = nid_model.predict(image, conf=0.5, verbose=False)[0]

    if len(result.boxes) == 0:
        return None

    boxes = result.boxes
    confs = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    
    bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)
    id_cropped = image[y1:y2, x1:x2].copy()

    return id_cropped


#Function to detect the national ID number using YOLO
def detect_national_id(id_cropped):

    results = digit_model.predict(id_cropped, conf=0.3, iou=0.5, verbose=False)[0]

    if len(results.boxes) == 0:
        return ""

    boxes     = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    confs     = results.boxes.conf.cpu().numpy()

    if len(class_ids) > 14:
        num14    = np.argsort(confs)[-14:]
        boxes    = boxes[num14]
        class_ids = class_ids[num14]

    sorted_idx = np.argsort(boxes[:, 0])
    return ''.join([str(int(class_ids[i])) for i in sorted_idx]) 


#Function to preprocess the cropped image for OCR
def preprocess_image(image):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_up  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray_up, None, h=7)
    enhanced = clahe.apply(denoised)
    binary   = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


#Function to convert Arabic digits to English digits
def convert_arabic_to_english(text):
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


#Function to extract national ID number using OCR
def extract_national_id(id_cropped):
    binary = preprocess_image(id_cropped)
    
    result = reader.readtext(
        binary,
        allowlist='0123456789٠١٢٣٤٥٦٧٨٩',
        detail=1,
        paragraph=False,
        decoder='beamsearch',
        beamWidth=10,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        text_threshold=0.2,
        low_text=0.2
    )
    
    ocr_digits = ""
    for box, text, _ in sorted(result, key=lambda x: x[0][0][0]):
        ocr_digits += re.sub(r'\D', '', convert_arabic_to_english(text))

    yolo_digits = detect_national_id(id_cropped)
   
    if len(yolo_digits) == 14:
        return yolo_digits

    elif len(ocr_digits) == 14:
        return ocr_digits

    else:
        return "Unverified"

'''full pipeline from image to national ID number'''
def extract_nid_from_image(input_img):

    img_array = np.frombuffer(input_img, np.uint8)
    image     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:                       
        return None, None, "Invalid Image"
 
    image_cropped = crop_card(image)
    
    if image_cropped is None:
        return image, None, "Card Not Detected"

    corrected_orientation = correct_orientation(image_cropped)

    corrected_skew = correct_skew(corrected_orientation)

    id_cropped = crop_id_box(corrected_skew)

    if id_cropped is None:
      return corrected_skew , None, "ID Not Detected"
    
    nid = extract_national_id(id_cropped)

    return corrected_skew, id_cropped, nid

