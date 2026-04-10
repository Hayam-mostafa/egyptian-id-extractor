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
    
    for angle in angles:
        if angle == 90:
            temp_img = cv2.rotate(image_cropped_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            temp_img = cv2.rotate(image_cropped_bgr, cv2.ROTATE_180)
        elif angle == 270:
            temp_img = cv2.rotate(image_cropped_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            temp_img = image_cropped_bgr

        img_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(img_rgb)

        if face_locations:
            return PImage.fromarray(img_rgb)
    
    return PImage.fromarray(cv2.cvtColor(image_cropped_bgr, cv2.COLOR_BGR2RGB))


'''correct the skew of the orientation corrected image'''
def correct_skew(corrected_orientation_img):
    img_gray = cv2.cvtColor(np.array(corrected_orientation_img), cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    angles = np.arange(-10, 10, 0.5)
    scores = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((img_bin.shape[1] // 2, img_bin.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(img_bin, M, (img_bin.shape[1], img_bin.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        hist = np.sum(rotated, axis=1)
        scores.append(np.var(hist))

    best_angle = angles[np.argmax(scores)]
    
    if abs(best_angle) < 0.3:
        return corrected_orientation_img

    M = cv2.getRotationMatrix2D((img_bin.shape[1] // 2, img_bin.shape[0] // 2), best_angle, 1)
    rotated_final = cv2.warpAffine(np.array(corrected_orientation_img), M, (img_bin.shape[1], img_bin.shape[0]),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return PImage.fromarray(rotated_final)

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


