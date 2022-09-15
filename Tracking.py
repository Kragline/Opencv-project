import cv2
import mediapipe as mp


class Tracking():
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=2)

    def get_pose_landmarks(self, img, draw=False):
        pose_landmarks_list = []
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        if results.pose_landmarks:
            for num, cors in enumerate(results.pose_landmarks.landmark):
                height, width = img.shape[:2]
                cor_x, cor_y = int(cors.x*width), int(cors.y*height)
                pose_landmarks_list.append([num, cor_x, cor_y])
                if draw == True:
                    self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return pose_landmarks_list

    def get_hand_landmarks(self, img, draw=False):
        hands_landmarks_list = []
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for one_hand in results.multi_hand_landmarks:
                for num, cors in enumerate(one_hand.landmark):
                    height, width = img.shape[:2]
                    cor_x, cor_y = int(cors.x*width), int(cors.y*height)
                    hands_landmarks_list.append([num, cor_x, cor_y])
                    if draw == True:
                        self.mp_draw.draw_landmarks(img, one_hand, self.mp_hands.HAND_CONNECTIONS)

        return hands_landmarks_list

    def get_face_landmarks(self, img, draw=False):
        face_landmarks_list = []
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face.process(rgb_img)

        if results.detections:
            for num, detection in enumerate(results.detections):
                height, width = img.shape[:2]
                det_bbox = detection.location_data.relative_bounding_box
                bbox = int(det_bbox.xmin * width), int(det_bbox.ymin * height), int(det_bbox.width * width), int(det_bbox.height * height)
                face_landmarks_list.append([bbox, detection.score])
                
                if draw == True:
                    self.mp_draw.draw_detection(img, detection)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        return face_landmarks_list

    def get_face_mesh(self, img, draw=False):
        face_mesh_list = []
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                if draw == True:
                        self.mp_draw.draw_landmarks(img, face, self.mp_face_mesh.FACEMESH_CONTOURS, self.draw_spec, self.draw_spec)

                for num, cors in enumerate(face.landmark):
                    height, width = img.shape[:2]
                    cor_x, cor_y = int(cors.x*width), int(cors.y*height)
                    face_mesh_list.append([num, cor_x, cor_y])
                    
        return face_mesh_list