import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
i = 0
while True:
    ret, frame = cap.read()
    
    # cv2.imshow("L", frame[:,0:int(640)])
    cv2.imshow("R", frame[:,int(640):])
    # cv2.imshow('frame', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('w'):
        i += 1
        # cv2.imwrite(f'data/two_calibration_image/left/{i}.jpg', frame[:,0:int(640)])
        # cv2.imwrite(f'data/two_calibration_image/right/{i}.jpg', frame[:,int(640):])
        # cv2.imwrite(f'data/one_calibration_image/{i}.jpg', frame[:,int(640):])
        cv2.imwrite(f'data/eye_hand_calibration_image/{i}.jpg', frame[:,int(640):])
        print(f"成功{i}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    