import cv2
import numpy as np

def load_and_enhance_image(image_path):
    """
    Load an image and apply advanced preprocessing techniques for optimal face detection.
    Handles different color spaces and enhances image quality.

    Args:
        image_path (str): Path to the input image

    Returns:
        tuple: (processed_gray_image, processed_color_image)
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Error: Image at {image_path} not found or cannot be loaded.")

    if len(image.shape) == 2:  # Grayscale
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            color_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:  # BGR
            color_image = image.copy()
    else:
        raise ValueError("Unsupported image format")

    lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    enhanced_color = cv2.fastNlMeansDenoisingColored(
        enhanced_color,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    enhanced_color = cv2.bilateralFilter(
        enhanced_color,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    gray_image = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)

    gray_image = clahe.apply(gray_image)

    blur = cv2.GaussianBlur(gray_image, (0, 0), 3)
    gray_image = cv2.addWeighted(gray_image, 1.5, blur, -0.5, 0)

    return gray_image, enhanced_color

def detect_face(cascade, gray_image, color_image):
    faces = cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        faces = cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

    return largest_face

def crop_and_resize_face(gray_image, face_coords, size=(200, 200)):
    x, y, w, h = face_coords
    face = gray_image[y:y+h, x:x+w]
    return cv2.resize(face, size)

def detect_rotation(face_1, face_2):
    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(face_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(face_2, None)

    if descriptors_1 is None or descriptors_2 is None:
        raise ValueError("Keypoints or descriptors not found in one of the faces.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    if len(points_1) >= 4 and len(points_2) >= 4:
        matrix, _ = cv2.estimateAffinePartial2D(points_1, points_2)
        if matrix is not None:
            rotation_angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
            return rotation_angle
    return 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def match_faces(face_1, face_2, good_match_threshold=30, distance_threshold=100):
    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(face_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(face_2, None)

    if descriptors_1 is None or descriptors_2 is None:
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < distance_threshold]

    return len(good_matches) > good_match_threshold, len(good_matches), good_matches, keypoints_1, keypoints_2

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image_1_path = '/Photos/parth_ac.jpg'
    image_2_path = '/Photos/parth_expjpg.jpg'

    try:
        gray_image_1, color_image_1 = load_and_enhance_image(image_1_path)
        gray_image_2, color_image_2 = load_and_enhance_image(image_2_path)

        face_coords_1 = detect_face(face_cascade, gray_image_1, color_image_1)
        face_coords_2 = detect_face(face_cascade, gray_image_2, color_image_2)

        face_1_resized = crop_and_resize_face(gray_image_1, face_coords_1)
        face_2_resized = crop_and_resize_face(gray_image_2, face_coords_2)

        rotation_angle = detect_rotation(face_1_resized, face_2_resized)
        if abs(rotation_angle) > 5:
            face_2_resized = rotate_image(face_2_resized, -rotation_angle)

        is_same_person, good_match_count, good_matches, keypoints_1, keypoints_2 = match_faces(face_1_resized, face_2_resized)

        if is_same_person:
            print(f"The images represent the same person. {good_match_count} good matches found.")
        else:
            print(f"The images do not represent the same person. Only {good_match_count} good matches found.")

        match_result = cv2.drawMatches(face_1_resized, keypoints_1, face_2_resized, keypoints_2, good_matches, None)
        cv2.imshow("Matched Faces", match_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "_main_":
    main()