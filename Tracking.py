import cv2
import mediapipe as mp


class Tracking():
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

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