import pyrebase
class PyrebaseConfig:
    def __init__(self):
        config = {"apiKey": "YOUR API KEY",
                    "authDomain": "xxx.firebaseapp.com",
                    "databaseURL": "https://xxx.firebaseio.com",
                    "storageBucket": "xxx.appspot.com",
                    "serviceAccount": "PATH OF ADMIN_SDK json FILE"}
        firebase = pyrebase.initialize_app(config= config)
        auth = firebase.auth()

        #Authenticate a User
        user = auth.sign_in_with_email_and_password("YOUR MAIL", "PASSWORD")

        #Id Token will be refresh in 1 hour.
        self.user = auth.refresh(user['refreshToken'])

        self.db = firebase.database()

    def postData(self, param):
        # Pass the user's id Token to the post method
        self.db.child("Warning").set(param, self.user['idToken'])