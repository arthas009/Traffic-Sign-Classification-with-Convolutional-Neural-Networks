import keras
import os
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from keras.models import load_model
import h5py
import tensorflow as tf
from PIL import Image
from skimage import transform
from skimage import data
import skimage as sm
from skimage import color
from skimage.color import rgb2gray
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np  

#tkinter app
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

# open a image folder
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
#load image to screen and make prediction with saved CNN model
def open_img():
    x = openfn()
    img = Image.open(x)
    img = img.resize((64, 64), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    # you must specify the path according to your own model path.
    model = load_model('../savedmodel/my_model.h5')
    testimage = load_img(x,target_size=(64,64))
    testimage = img_to_array(testimage)
    testimage = np.expand_dims(testimage, axis=0)
    prediction = model.predict(testimage)
    #print the prediction
    print(np.argmax(prediction[0]))
    var = StringVar()
    label = Label(root, textvariable=var, relief=RAISED )
    # give the output to screen
    var.set("This photo belongs to "+str(np.argmax(prediction[0]))+". class")
    label.pack()
    
btn = Button(root, text='open image', command=open_img).pack()

root.mainloop()