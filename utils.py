from PIL import Image as PImage
from ultralytics import YOLO
import face_recognition
import easyocr
import cv2
import numpy as np
import re

'''load Models'''
card_model   = YOLO('Models/card_detector.pt')
digit_model = YOLO('Models/digit_detector.pt')
id_model     = YOLO('Models/nid_detector.pt')
reader       = easyocr.Reader(['ar'], gpu=False)
clahe       = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) 


'''detect and crop the card from the input image'''
def crop_card(image,card_model):
    prediction_results = card_model.predict(image, conf=0.5, iou=0.45)
    prediction_result  = prediction_results[0]

    if len(prediction_result.boxes) == 0:
        raise ValueError("No card detected")

    boxes = prediction_result.boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)

    box = boxes.xyxy[best_idx].cpu().numpy()
    x1, y1, x2, y2 = box.astype(int)

    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    image_cropped = image[y1:y2, x1:x2].copy()
       
    return image_cropped


'''correct the orientation of the cropped card'''
def correct_orientation(image_cropped_bgr):
    angles = [0, 90, 180, 270]
    best_angle = 0
    max_conf = 0
    
    temp_bgr = image_cropped_bgr.copy()
    
    for angle in angles:
        if angle == 90:
            rotated = cv2.rotate(temp_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(temp_bgr, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(temp_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = temp_bgr
 
        results = id_model.predict(rotated, conf=0.4, verbose=False)[0]
        
        if len(results.boxes) > 0:
            current_conf = results.boxes.conf.cpu().numpy().max()
            if current_conf > max_conf:
                max_conf = current_conf
                best_angle = angle

    if best_angle == 90:
        final_img = cv2.rotate(temp_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif best_angle == 180:
        final_img = cv2.rotate(temp_bgr, cv2.ROTATE_180)
    elif best_angle == 270:
        final_img = cv2.rotate(temp_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        final_img = temp_bgr
        
    print(f" Orientation Fixed: {best_angle}° (Confidence: {max_conf:.2f})")
    
    img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    from PIL import Image as PImage
    return PImage.fromarray(img_rgb)


'''correct the skew of the orientation corrected image'''
def correct_skew(image):

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    median_angle = np.median(angles)

    if abs(median_angle) > 20:
        return image

    h, w = gray.shape

    rad = np.deg2rad(median_angle)
    new_w = int(abs(w * np.cos(rad)) + abs(h * np.sin(rad)))
    new_h = int(abs(h * np.cos(rad)) + abs(w * np.sin(rad)))

    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        img,M,(new_w, new_h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    return PImage.fromarray(rotated)


'''Expand Area'''
def expand_bbox_height(bbox, scale=1.5, image_shape=None):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [x1, new_y1, x2, new_y2]


'''detect and crop the ID region from the corrected image'''
def crop_id_box(img_bgr, id_model, pad=5):
    result = id_model.predict(img_bgr, conf=0.5, verbose=False)[0]

    if len(result.boxes) == 0:
        raise ValueError("ID region not detected")

    boxes = result.boxes
    confs = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    bbox = expand_bbox_height(bbox, scale=1.3, image_shape=img_bgr.shape)

    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(img_bgr.shape[1], x2+pad), min(img_bgr.shape[0], y2+pad)
    id_cropped = img_bgr[y1:y2, x1:x2].copy()

    return id_cropped


def detect_national_id(id_cropped):
    gray = cv2.cvtColor(id_cropped, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results = digit_model.predict(gray_3ch, conf=0.3, iou=0.5, verbose=False)[0]

    if len(results.boxes) == 0:
        return ""

    boxes     = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    confs     = results.boxes.conf.cpu().numpy()

    if len(class_ids) > 14:
        top14    = np.argsort(confs)[-14:]
        boxes    = boxes[top14]
        class_ids = class_ids[top14]

    sorted_idx = np.argsort(boxes[:, 0])
    return ''.join([str(int(class_ids[i])) for i in sorted_idx])

def preprocess_image(image):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_up  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray_up, None, h=11)
    enhanced = clahe.apply(denoised)
    blurred  = cv2.GaussianBlur(enhanced, (3,3), 0)
    binary   = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    return binary

def convert_arabic(text):
     return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


'''extract national ID number using OCR'''
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

    
    all_digits = ""
    for box, text, conf in sorted(result, key=lambda x: x[0][0][0]):
        all_digits += re.sub(r'\D', '', convert_arabic(text))

    yolo_digits = detect_national_id(id_cropped)

    print(f"  OCR  : {all_digits} ({len(all_digits)}/14)")
    print(f"  YOLO : {yolo_digits} ({len(yolo_digits)}/14)")

    if all_digits == yolo_digits and len(all_digits) == 14:
        print("Perfect Match! → Using YOLO")
        return yolo_digits

    if len(all_digits) == 14 and len(yolo_digits) == 14:
        matches = sum(a == b for a, b in zip(all_digits, yolo_digits))
        print(f"Both detected ({matches}/14 match) → Using YOLO")
        return yolo_digits

    if len(all_digits) < 14 and len(yolo_digits) == 14:
        print("OCR incomplete → YOLO")
        return yolo_digits
    
    if len(all_digits) > 14 and len(yolo_digits) == 14:
        print("OCR overdetected → YOLO")
        return yolo_digits

    if len(all_digits) == 14 and len(yolo_digits) < 14:
        print("YOLO incomplete → OCR")
        return all_digits

    print(f"Failed | OCR={len(all_digits)} | YOLO={len(yolo_digits)}")
    
    return None



'''full pipeline from image to national ID number'''
def full_pipeline(input_img):

    img_array = np.frombuffer(input_img, np.uint8)
    image     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    image_cropped = crop_card(image, card_model)

    corrected_orientation = correct_orientation(image_cropped)

    corrected_skew = correct_skew(corrected_orientation)

    corrected_skew_np = np.array(corrected_skew)
    corrected_skew_bgr = cv2.cvtColor(corrected_skew_np, cv2.COLOR_RGB2BGR)

    id_cropped = crop_id_box(corrected_skew_bgr, id_model)
    nid = extract_national_id(id_cropped)

    return corrected_skew, id_cropped, nid
