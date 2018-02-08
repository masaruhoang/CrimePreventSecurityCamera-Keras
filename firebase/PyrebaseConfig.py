import pyrebase
class PyrebaseConfig:
    def __init__(self):
        config = {"apiKey": "AIzaSyCYC1lknF0lSo6rBNWEb75z5fZ--WM1A4w",
                    "authDomain": "direct-raceway-141000.firebaseapp.com",
                    "databaseURL": "https://direct-raceway-141000.firebaseio.com",
                    "storageBucket": "direct-raceway-141000.appspot.com",
                    "serviceAccount": "D:/IoT/NeuralNetwork/python/AutonomousDrivingAndy/firebaseAdminKey"
                                      +"/direct-raceway-141000-firebase-adminsdk-cyiaw-58e09d1493.json"}
        firebase = pyrebase.initialize_app(config= config)
        auth = firebase.auth()

        #Authenticate a User
        user = auth.sign_in_with_email_and_password("rinrin1992dn@gmail.com", "hoangthehuy123456789")

        #Id Token will be refresh in 1 hour.
        self.user = auth.refresh(user['refreshToken'])

        self.db = firebase.database()

    def postData(self, param):
        # Pass the user's id Token to the post method
        self.db.child("Warning").set(param, self.user['idToken'])