import cv2
import mediapipe as mp


# class which creates face mesh


class FaceMesh():
    def __init__(self, staticMode=False, maxFaces=1, detConfidence=0.5, trackConfidence=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.detConfidence = detConfidence
        self.trackConfidence = trackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.detConfidence, self.trackConfidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    # creates the face mesh
    def makeFaceMesh(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.drawSpec
                    )
                face = []
                for lm in face_landmarks.landmark:
                    ih, iw = img.shape[:2]
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append([face])
        return faces[0] if len(faces)>0 else None


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        reading, img = cap.read()
        img = cv2.flip(img, 1)
        if not reading:
            print("Ignoring empty frame")
            continue
        face_mesh = FaceMesh(
            staticMode=False,
            maxFaces=1,
            detConfidence=0.5,
            trackConfidence=0.5
        )
        face_mesh.makeFaceMesh(img, True)
        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
