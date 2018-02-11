import cv2, time
from firebase.PyrebaseConfig import *
import GmailConfig as gm
import HandGestureCNN as my_cnn

'''======================= PARAMS ============================'''
'''==========================================================='''

# Coordination of ROI Image. It will help display a rectangle on the Screen.
x0 = 400
y0 = 200
height = 200
width = 200

lastgesture = -1

# Counter which count Image is saved to data_set directory
counter = 0

# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = "./data_set/"
mod = 0

# =====================================
im_save_state = False
check_state_prediction = False
check_training_mode = False

# Has any weapon or violence' reaction check state
has_violence_weapon = 0
has_money = 0
# Prediction Threshold be able to filter output.
min_predict_val = 70

'''=========================== SAVE ROI IMAGE ======================='''
'''=================================================================='''
def saveROIImg(img):
    global counter, gestname, path, im_save_state
    if counter > (numOfSamples - 1):
        # Reset the parameters
        im_save_state = False
        gestname = ''
        counter = 0
        chk_next_class = input("Do you want save the next class(y/n): ")
        if chk_next_class == 'y':
            im_save_state = True
            gestname = input("Type name for next class: ")
        return
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:", name)
    cv2.imwrite(path + name + ".png", img)
    time.sleep(0.04)

'''================= CROP AND CONVERT TO GRAYSCALE ==================='''
'''==================================================================='''
def binaryMask(frame, x0, y0, width, height, pyrebase):
    global check_state_prediction, check_training_mode, mod, lastgesture, im_save_state, has_violence_weapon, has_money

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), 2)
    roi = frame[y0:y0 + height, x0:x0 + width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, min_predict_val, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if im_save_state == True:
        saveROIImg(res)
    elif check_state_prediction == True:
        predicted_label = my_cnn.predict(mod, res)
        cv2.putText(frame, "["+str(predicted_label)+"]", (x0+50, y0-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, 1)

        # Dangerous Weapon or Violent reaction
        if(predicted_label == "PUNCH" or predicted_label == "KNIFE"):
            has_violence_weapon += 1
            has_money = 0
            if(has_violence_weapon == 17): #Will send Warning's data to Firebase and Gmail's Store owner
                pyrebase.postData("Violence")
                has_violence_weapon = 0
                gm.start_send_mail("KNIFE")

        elif(predicted_label == "HASMONEY"):
            has_money += 1
            has_violence_weapon = 0
            if(has_money == 17):
                pyrebase.postData("Hasmoney")
                has_money = 0
                gm.start_send_mail("HASMONEY")

    return res


'''========================================================================================'''
'''====================================== MAIN ============================================'''
'''========================================================================================'''
def Main():
    global check_state_prediction, check_training_mode, mod, binaryMode, \
           x0, y0, width, height, im_save_state, gestname, path

    #Init Pyrebase
    pyrebase = PyrebaseConfig()

    #Params for put text on the screen
    font = cv2.FONT_HERSHEY_DUPLEX
    size = 0.5
    fx = 400
    fy = 50
    fh = 18


    ## Grab camera input
    cap = cv2.VideoCapture(1)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    while (True):
        ret, frame = cap.read()

        frame = cv2.flip(frame, 3)

        if ret == True:
            binaryMask(frame, x0, y0, width, height, pyrebase)
        cv2.line(frame, (370,0), (370,480), (0,0,0), 2)
        cv2.putText(frame, 'Opts-Press key from keyboard', (fx-20, fy), font, 0.5, (0, 0, 0), 1, 1)
        cv2.putText(frame, 't - Training', (fx, fy + fh), font, size, (0, 50, 0), 1, 1)
        cv2.putText(frame, 'c - Image Collection', (fx, fy+ 2*fh), font, size, (0,50,0),1, 1)
        cv2.putText(frame, 'p - Prediction',(fx, fy + 3*fh), font, size, (0, 50, 0), 1, 1)
        cv2.putText(frame, 'esc - Quit', (fx, fy +  4*fh), font, size, (0, 50, 0), 1, 1)
        cv2.putText(frame, '[Counter Area]',(fx+50, fy+ 21*fh), font, size, (0,0,0),2,1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Original', gray)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        ## Use g key to start gesture predictions via CNN
        if key == ord('p'):
            print("Prediction Mode - {}".format(check_state_prediction))
            mod = my_cnn.load_cnn(0)
            check_state_prediction = not check_state_prediction

        ## Use c key to start collect image and save to data_set
        if key == ord('c'):
            gestname = input("Type name of the classes (Ex: PUNCH, KNIFE....): ")
            im_save_state = not im_save_state

        ## Use t key to start training mode
        if key == ord('t'):
            mod = my_cnn.load_cnn(-1)
            my_cnn.trainModel(mod)
            check_training_mode = not check_training_mode

        ## Use Esc key to close the program
        elif key == 27:
            break


    # Realse & destroy
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()

