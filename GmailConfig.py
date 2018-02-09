import smtplib
import threading


def send_mail(subject, msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login("GMAIL ID", "PASSWORD")
        message = 'Subject: {}\n\n{}'.format(subject, msg)
        server.sendmail("SEND TO MAIL", "SEND TO MAIL", message)
        server.quit()
        print("Email Sent!!!")
        print("Subject: " + subject)
        print("Email Sent!!! With content: " + msg)

    except:
        print("Sending Email is Failed.")


def start_send_mail(state):
    global subject, msg
    if state == "PUNCH" or state == "KNIFE":
        subject = "[Warning!!!] Maybe has any violence reaction or dangerous weapon from Hand's customer'."
        msg = "Hand gesture's Violence Or Dangerous Weapon have been detected. Be careful!!!!! "
    elif state == "HASMONEY":
        subject = "[Warning!!!] Maybe someone took money out from register counter."
        msg = "Hand gesture's Steal have been detected. Be carefull!!!!! "

    t = threading.Thread(name="GMAIL_SENDING", target=send_mail(subject, msg, ))
    t.start()
    t.join()
