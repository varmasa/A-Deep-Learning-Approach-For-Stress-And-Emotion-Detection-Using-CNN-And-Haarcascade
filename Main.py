from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
from tkinter import Tk, Text, Scrollbar, Button, Label, ttk, filedialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread, Lock
from collections import Counter
import tkinter as tk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import cv2
import seaborn as sns
# from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import soundfile
import librosa
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

cascPath = "C:\\Users\\Saivarma\\Desktop\\cp\\StressDetection\\model\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
face_emotion = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
stress_detection_results = []
face_emotions_detected = []

main = tkinter.Tk()
main.title("A DEEP LEARNING APPROACH FOR STRESS AND EMOTION DETECTION USING CNN AND HAAR CASCADE") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y, rf_model, tfidf_vectorizer
global face_classifier
global speech_X, speech_Y
global speech_classifier
global accuracy, precision, recall, fscore
global speech_X_train, speech_X_test, speech_y_train, speech_y_test
global image_X_train, image_X_test, image_y_train, image_y_test
stop_words = set(stopwords.words('english'))

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        
    
def uploadDataset():
    global filename, tfidf_vectorizer
    filename = filedialog.askdirectory(initialdir=".")
    f = open('model/tfidf.pckl', 'rb')
    tfidf_vectorizer = pickle.load(f)
    f.close()  
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
def processDataset():
    text.delete('1.0', END)
    global X, Y, text_X, text_Y
    global speech_X, speech_Y
    global speech_X_train, speech_X_test, speech_y_train, speech_y_test
    global image_X_train, image_X_test, image_y_train, image_y_test
    global text_X_train, text_X_test, text_y_train, text_y_test
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        speech_X = np.load('model/speechX.txt.npy')
        speech_Y = np.load('model/speechY.txt.npy')
        text_X = np.load("model/textX.txt.npy")
        text_Y = np.load("model/textY.txt.npy")
        indices = np.arange(text_X.shape[0])
        np.random.shuffle(indices)
        text_X = text_X[indices]
        text_Y = text_Y[indices]
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    Y.append(getID(name))        
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    image_X_train, image_X_test, image_y_train, image_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total number of images found in dataset are  : "+str(len(X))+"\n")
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% images used to train Deep Learning Algorithm : "+str(image_X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test Deep Learning Algorithm : "+str(image_X_test.shape[0])+"\n")
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.1)

def datasetImages():
    # size of the image: 48*48 pixels
    pic_size = 48

    # input path for the images
    base_path = "C:\\Users\\Saivarma\\Desktop\\cp\\StressDetection\\Dataset\\"

    plt.figure(0, figsize=(12,20))
    cpt = 0
    for expression in os.listdir(base_path + "Images/"):
        for i in range(1,6):
            cpt = cpt + 1
            plt.subplot(7,5,cpt)
            img = load_img(base_path + "Images/" + expression + "/" +os.listdir(base_path + "Images/" + expression)[i], target_size=(pic_size, pic_size))
            plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.show()

def trainingImages():
    emotions = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    image_counts = [3995, 436, 4097, 7215, 4965, 4830, 3171]

    # Create a color palette with a distinct color for each emotion
    palette = sns.color_palette("hsv", len(emotions))

    # Create the barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotions, y=image_counts, palette=palette)

    # Set the title and labels for the plot
    plt.title('Images Used For Model Training')
    plt.xlabel('Facial Emotions')
    plt.ylabel('Number of Images')

    # Show the plot
    plt.show()

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    class_report = classification_report(y_test, predict)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy   :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision  :  "+str(p)+"\n")
    text.insert(END,algorithm+" Recall     :  "+str(r)+"\n")
    text.insert(END,algorithm+" F1Score    :  "+str(f)+"\n\n")
    text.insert(END,algorithm+" Classification Report:\n" + class_report + "\n\n")
    #text.update_idletasks()
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predict)
    
    # Plotting the confusion matrix
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(cm, annot=True, cmap="viridis", fmt="g") 
    plt.title('Confusion Matrix') 
    plt.xlabel('Predicted Labels') 
    plt.ylabel('True Labels') 
    plt.xticks(rotation=90)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5) 
    plt.tight_layout()
    plt.show()
    
def trainFaceCNN():
    global face_classifier, accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    global image_X_train, image_X_test, image_y_train, image_y_test
    text.delete('1.0', END)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            face_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        face_classifier.load_weights("model/cnnmodel_weights.h5")
        #face_classifier._make_predict_function()                  
    else:
        face_classifier = Sequential()
        face_classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Flatten())
        face_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        face_classifier.add(Dense(output_dim = 7, activation = 'softmax'))
        face_classifier.summary()
        face_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = face_classifier.fit(image_X_train, image_y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        face_classifier.save_weights('model/cnnmodel_weights.h5')            
        model_json = face_classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = face_classifier.predict(image_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    calculateMetrics("CNN Image Algorithm", predict, y_test1)   

def predictFacialStress():
    global face_classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = face_classifier.predict(img)
    predict = np.argmax(preds)
    output = "Stressed"
    if predict == 3 or predict == 4:
        output = "Non Stressed"    
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Facial Output : '+output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    #cv2.putText(img, 'Percentage : '+str(0.7), (75, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)
    cv2.imshow('Facial Output : '+output, img)
    cv2.waitKey(0)

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("Already started!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
        
def runWebCam():
    global face_classifier
    global stress_detection_results
    global face_emotions_detected
    video_capture = VideoCaptureAsync().start()
    while True:
        frame = video_capture.read()
        # Keep the frame at its original size
        # frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        print("Found {0} faces!".format(len(faces)))

        for (x, y, w, h) in faces:
            sub_face = frame[y:y+h, x:x+w]
            sub_face = cv2.resize(sub_face, (32, 32))  # This resize is for the neural network input, not for display
            im2arr = np.array(sub_face).reshape(1, 32, 32, 3).astype('float32') / 255
            preds = face_classifier.predict(im2arr)
            predict = np.argmax(preds)
            output = "Non Stressed" if predict in [3, 4] else "Stressed"
            stress_detection_results.append(output)
            face_emotions_detected.append(face_emotion[predict])
            cv2.putText(frame, 'Facial Expression Recognized as : ' + output, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, 'Stress Feelings Recognized as   : ' + face_emotion[predict], (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()
    cv2.destroyAllWindows()

def graph():
    global accuracy, precision, recall, fscore

    # Metrics and their values
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy[0], precision[0], recall[0], fscore[0]]

    # Medium-tone colors for the bars
    colors = ['skyblue', 'mediumseagreen', 'coral', 'mediumpurple']

    # Setting up the histogram-style plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors)

    # Adding text labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom')  # va: vertical alignment

    # Setting the title and labels
    plt.title('CNN Image Algorithm Performance Metrics', pad=20, fontsize=16, y=1.05)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Values (%)', fontsize=14)
    plt.ylim(0, 100)  # Assuming the values are percentages

    # Display the plot
    plt.show()

def results():
    global stress_detection_results
    global face_emotions_detected
    global tips, website_links, youtube_links, song_links

    # Calculate the most common stress result and emotion
    stress_result_count = Counter(stress_detection_results)
    emotion_count = Counter(face_emotions_detected)

    if stress_result_count:
        most_common_stress = stress_result_count.most_common(1)[0][0]
    else:
        most_common_stress = "No data"

    if emotion_count:
        most_common_emotion = emotion_count.most_common(1)[0][0]
    else:
        most_common_emotion = "No data"
    
    # Get the recommendations based on the most common emotion detected
    emotion_tips = tips.get(most_common_emotion, ["No recommendations"])
    emotion_websites = website_links.get(most_common_emotion, ["No links available"])
    emotion_youtube = youtube_links.get(most_common_emotion, ["No links available"])
    emotion_songs = song_links.get(most_common_emotion, ["No links available"])

    # Update the text widget with the results and recommendations
    text.delete('1.0', END)
    text.insert(END, f"Stress Detection Results:\n{most_common_stress}\n\n")
    if most_common_stress == "Stressed":
        text.insert(END, "Recommendations for managing stress:\n")
        for tip in tips["Stressed"]:
            text.insert(END, f"- {tip}\n")
    elif most_common_stress == "Non Stressed":
        text.insert(END, "Recommendations for maintaining a non-stressed state:\n")
        for tip in tips["Non Stressed"]:
            text.insert(END, f"- {tip}\n")
    else:
        text.insert(END, "No stress data available to provide recommendations.\n")
    text.insert(END, f"\nStress Feelings Results:\n{most_common_emotion}\n\n")
    text.insert(END, f"Recommendations for {most_common_emotion} Feelings:\n")

    # Append tips, website links, youtube links, and songs for the most common emotion
    for tip in emotion_tips:
        text.insert(END, f"- {tip}\n")
    text.insert(END, "\nHelpful Links:\n")
    for link in emotion_websites + emotion_youtube + emotion_songs:
        text.insert(END, f"- {link}\n")

def exit():
    main.destroy()

tips = {
    "Stressed": [
        "Take deep breaths and focus on breathing slowly and calmly.",
        "Engage in physical activity, such as a brisk walk or a run, to help reduce stress hormones.",
        "Try mindfulness meditation or yoga to help bring a sense of calm.",
        "Break down big tasks into smaller, manageable steps and take them one at a time.",
        "Reach out to a friend or family member to talk about what you're experiencing."
    ],
    "Non Stressed": [
        "Maintain a regular exercise routine to keep stress at bay.",
        "Ensure you get plenty of sleep, aiming for 7-9 hours each night.",
        "Spend time on hobbies and activities that you enjoy and find relaxing.",
        "Practice gratitude by reflecting on or writing down things you are thankful for each day.",
        "Stay connected with your social network to foster a sense of belonging and support."
    ],
    "fear": ["Drink water", "Get a good night's sleep", "Eat wholesome meals", "Go for a walk", "Turn off news feed/social media", "Talk to someone"],
    "angry": ["Repeat gentle phrases to yourself", "Take a walk", "Use visualization to calm down", "Focus on your breathing", "Phone a friend", "Watch a stand up comedy"],
    "sad": ["Do things you enjoy (or used to)", "Get quality exercise", "Eat a nutritious diet", "Challenge negative thinking"],
    "happy": ["Savor the moment", "Share your happiness with others", "Keep a gratitude journal", "Do something kind for someone else"],
    "disgust": ["Reflect on the cause", "Express your feelings in a journal", "Talk about it with someone you trust", "Distract yourself with a hobby"],
    "surprised": ["Take a moment to process", "Share your surprise with a friend", "Write down how you feel", "Enjoy the moment, if it's a positive surprise"],
    "neutral": ["Explore a new interest", "Engage in mindfulness or meditation", "Take a leisurely walk", "Read a book or watch a movie"]
}
website_links = {
    "fear": ["https://www.businessinsider.in/science/health/heres-how-to-take-care-of-yourself-if-youre-feeling-scared-or-sad-right-now/articleshow/55342883.cms", "https://mhanational.org/what-can-i-do-when-im-afraid"],
    "angry": ["https://www.thehotline.org/resources/how-to-cool-off-when-youre-angry/", "https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/anger-management/art-20045434"],
    "sad": ["https://www.vandrevalafoundation.com/", "https://www.healthline.com/health/depression/recognizing-symptoms#fatigue"],
    "happy": ["https://www.actionforhappiness.org/", "https://www.verywellmind.com/how-to-be-happy-ways-to-be-happier-3144765"],
    "disgust": ["https://www.yourtango.com/experts/billmaiermsw/what-do-when-you-feel-absolute-disgust-toward-others-affects-your-mental-health", "https://psyche.co/ideas/you-can-train-yourself-to-find-disgusting-things-less-gross"],
    "surprised": ["https://www.psychologytoday.com/us/blog/fulfillment-any-age/202204/the-unexplored-emotion-surprise", "https://www.wnycstudios.org/podcasts/takeaway/segments/surprise-unexpected-why-it-feels-good-and-why-its-good-us"],
    "neutral": ["https://www.buddhistinquiry.org/article/what-about-neutral-feelings/", "https://braincleanupcoach.com/the-value-of-neutral-feelings/"]
}
youtube_links = {
    "fear": ["https://www.youtube.com/watch?v=IAODG6KaNBc"],
    "angry": ["https://www.youtube.com/watch?v=P6aPg3YBvBQ"],
    "sad": ["https://www.youtube.com/watch?v=P6aPg3YBvBQ"],
    "happy": ["https://www.youtube.com/watch?v=y6Sxv-sUYtM"], 
    "disgust": ["https://youtu.be/gkI551rjVzc?si=cdLhJlXIbwrstGnV"],
    "surprised": ["https://youtu.be/Cc7-61O6TtE?si=9ZHCH61V5tAoBzAO"],
    "neutral": ["https://youtu.be/coev3dg3t8I?si=02sc4JcvEN3pKb3H"]
}
song_links = {
    "fear": ["https://www.youtube.com/watch?v=GyA8ccqwp-4&feature=youtu.be", "https://www.bing.com/videos/search?q=alone+part+2&docid=607990227673701963&mid=1B6860319511BF2C5CC21B6860319511BF2C5CC2&view=detail&FORM=VIRE"],
    "angry": ["https://www.youtube.com/watch?v=e74wLJ_KRes&feature=youtu.be", "https://www.youtube.com/watch?v=JwWz-94a_Hk&feature=youtu.be"],
    "sad": ["https://www.youtube.com/watch?v=25ROFXjoaAU&feature=youtu.be", "https://www.youtube.com/watch?v=BzE1mX4Px0I"],
    "happy": ["https://www.youtube.com/watch?v=vGZhMIXH62M", "https://www.youtube.com/watch?v=Pkh8UtuejGw"],
    "disgust": ["https://youtu.be/bQ2UgBQuXis?si=q8wiaYb-hG9UAVIX"],
    "surprised": ["https://youtu.be/knZq-si0a7E?si=Hdm2bliL0OuEOTZk"],
    "neutral": ["https://youtu.be/ngORmvyvAaI?si=KuUVeUeFYUgZxpog"]
}

font = ('times', 13, 'bold')
title = Label(main, text='A Deep Learning Approach For Stress And Emotion Detection Using CNN And HAAR CASCADE')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=420,y=100)
text.config(font=font1)

font1 = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Stress Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1)

processButton = Button(main, text="Dataset Images", command=datasetImages)
processButton.place(x=50,y=200)
processButton.config(font=font1) 

processButton = Button(main, text="Training Images", command=trainingImages)
processButton.place(x=50,y=250)
processButton.config(font=font1)

cnnButton = Button(main, text="Facial Stress CNN Algorithm", command=trainFaceCNN)
cnnButton.place(x=50,y=300)
cnnButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

predictfaceButton = Button(main, text="Predict Facial Stress", command=predictFacialStress)
predictfaceButton.place(x=50,y=400)
predictfaceButton.config(font=font1)

predictfaceButton = Button(main, text="Facial Stress from Cam", command=runWebCam)
predictfaceButton.place(x=50,y=450)
predictfaceButton.config(font=font1)

moodButton = Button(main, text="Web Cam Results And Recommendations", command=results)
moodButton.place(x=50, y=500)
moodButton.config(font=font1)

ExitButton = Button(main, text="Close GUI", command=exit)
ExitButton.place(x=50,y=550)
ExitButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()