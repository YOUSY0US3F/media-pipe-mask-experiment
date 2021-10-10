import FindFaceMesh
import cv2
import numpy as np
import gif2numpy

path = "Images/lfdna5hce6x31.gif"


# uses homography to project image onto landmarks in the frame
def stickImage(frame, img, landmarks, index_1, index_2, index_3, index_4, face=0):
    h1, w1 = img.shape[:2]
    h2, w2 = frame.shape[:2]
    pts1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    pts2 = np.float32(
        [landmarks[face][index_1], landmarks[face][index_2],
         landmarks[face][index_3], landmarks[face][index_4]]
    )
    # print(f"index_3: {landmarks[face][index_3]} index_4: {landmarks[face][index_4]}")
    if landmarks[face][index_4][0] >= landmarks[face][index_3][0] or landmarks[face][index_1][0] >= \
            landmarks[face][index_2][0] or landmarks[face][index_1][1] >= landmarks[face][index_4][1] or \
            landmarks[face][index_2][1] >= landmarks[face][index_3][1]:
        return None, None
    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    im1Reg = cv2.warpPerspective(img, h, (w2, h2))
    mask2 = np.zeros(frame.shape, dtype=np.uint8)
    mask_color = (255,) * frame.shape[2]
    cv2.fillConvexPoly(mask2, np.int32(pts2), mask_color)
    return im1Reg, mask2


def buildMask(frame, img, landmarks):
    frame.flags.writeable = True
    # positions where images will go
    squares = [[116, 47, 203, 147], [6, 277, 423, 1],
               [47, 6, 1, 203], [277, 345, 376, 423],
               [203, 423, 415, 191], [147, 203, 191, 192],
               [423, 376, 416, 415], [192, 191, 181, 169],
               [409, 416, 364, 405], [90, 320, 421, 201],
               [169, 181, 201, 170], [405, 364, 395, 421],
               [201, 421, 377, 148], [170, 201, 148, 176],
               [421, 395, 400, 377], [143, 6, 47, 116],
               [6, 372, 345, 277], [55, 285, 351, 122],
               [103, 332, 285, 55], [70, 55, 193, 124]]
    master_image = np.zeros(frame.shape, dtype=np.uint8)
    master_mask = np.zeros(frame.shape, dtype=np.uint8)
    # make the mouth
    image, mask = buildMouth(frame, landmarks)
    if image is not None:
        master_image = cv2.bitwise_or(image, master_image)
        master_mask = cv2.bitwise_or(mask, master_mask)
    # make the eyes
    image, mask = buildEyes(frame,landmarks)
    if image is not None:
        master_image = cv2.bitwise_or(image, master_image)
        master_mask = cv2.bitwise_or(mask, master_mask)
    for square in squares:
        image, mask = stickImage(frame, img, landmarks, square[0], square[1], square[2], square[3])
        if image is not None:
            master_image = cv2.bitwise_or(image, master_image)
            master_mask = cv2.bitwise_or(mask, master_mask)
    master_mask = cv2.bitwise_not(master_mask)
    masked_frame = cv2.bitwise_and(frame, master_mask)
    cv2.imshow('mask', master_mask)
    cv2.imshow('image', master_image)
    return cv2.bitwise_or(master_image, masked_frame)


def buildMouth(frame, landmarks):
    image = cv2.imread("Images/LED.jpg")
    lips = []
    markers = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402,
               317, 14, 87, 178, 88, 95]
    for marker in markers:
        lips.append([landmarks[0][marker][0], landmarks[0][marker][1]])
    lips = np.int32([lips])
    pic, mask = stickImage(frame, image, landmarks, 92, 322, 424, 194)
    if pic is None:
        return None, None
    lines = np.zeros(frame.shape, dtype=np.uint8)
    lines = cv2.fillPoly(lines, lips,
                          (255, 129, 139))
    pic = cv2.bitwise_xor(lines, pic)
    return pic, mask

def buildEyes(frame, landmarks):
    image = cv2.imread("Images/LED.jpg")
    leye = []
    reye = []
    master = np.zeros(frame.shape, dtype=np.uint8)
    eyes = np.zeros(frame.shape, dtype=np.uint8)
    mask = np.zeros(frame.shape, dtype=np.uint8)
    left_markers = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144,
               163, 7]
    right_markers = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380,
                     381, 382]
    left_pic, left_mask = stickImage(frame, image, landmarks, 46, 55, 122, 228)
    right_pic, right_mask = stickImage(frame, image, landmarks, 285, 276, 448, 343)
    if left_pic is None and right_pic is None:
        return None, None
    if left_pic is not None:
        for marker in left_markers:
            leye.append([landmarks[0][marker][0], landmarks[0][marker][1]])
        leye = np.int32([leye])
        master = cv2.bitwise_or(left_pic, master)
        eyes = cv2.fillPoly(eyes, leye,
                            (255, 129, 139))
        mask = cv2.bitwise_or(left_mask, mask)
    if right_pic is not None:
        for marker in right_markers:
            reye.append([landmarks[0][marker][0], landmarks[0][marker][1]])
        reye = np.int32([reye])
        master = cv2.bitwise_or(right_pic, master)
        eyes = cv2.fillPoly(eyes, reye,
                            (255, 129, 139))
        mask = cv2.bitwise_or(right_mask, mask)
    master = cv2.bitwise_xor(eyes,master)



    return master, mask

def main():
    img = None
    count = 0
    frames = None
    if path[-3:] == "gif":
        frames, exts, image_specs = gif2numpy.convert(path)
        img = frames[count]
        print("read gif")
    else:
        img = cv2.imread(path)
    h1, w1 = img.shape[:2]
    img = cv2.rectangle(img, (0, 0), (w1, h1), (0, 0, 0), 20)
    cap = cv2.VideoCapture(0)
    reading = None
    frame = None
    face_mesh = FindFaceMesh.FaceMesh(
        staticMode=False,
        maxFaces=1,
        detConfidence=0.5,
        trackConfidence=0.5
    )
    faces = None
    while cap.isOpened():
        if frames is not None:
            img = frames[count]
            img = cv2.rectangle(img, (0, 0), (w1, h1), (0, 0, 0), 20)
        reading, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        if not reading:
            print("Ignoring empty frame")
            continue
        faces = face_mesh.makeFaceMesh(frame, False)
        if faces is not None:
            frame = buildMask(frame, img, faces)
        cv2.imshow('Webcam', frame)
        if frames is not None:
            count += 1
            if count >= len(frames):
                count = 0
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()


main()
