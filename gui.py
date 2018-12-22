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

names = ["Kasisli Yol","Kasisli Köprü Yaklaşımı","Kaygan Yol","Sola Tehlikeli Viraj","Sağa Tehlikeli Viraj",
         "Sola Tehlikeli Devamlı Virajlar","Sağa Tehlikeli Devamlı Virajlar","Okul Yolu","Bisikletli Geçebilir","Ehli Hayvan Geçebilir","Yol Çalışması","Işıklı İşaret Cihazı"
         ,"Kontrollü Demiryolu Geçidi","Dikkat","Her İki Taraftan Daralan Kaplama","Soldan Daralan Kaplama","Sağdan Daralan Kaplama"
         ,"Ana Yol - Tali Yol Kavşağı","Kontrolsüz Kavşak","Yol Ver","Karşıdan Gelene Yol Ver","Dur","Girişi Olmayan Yol"
         , "Bisiklet Giremez","Tanımsız Tabela","Kamyon Giremez","Genişliği 2,10 mt. den Fazla Giremez"
         ,"Yüksekliği ^ ^ mt. den Fazla Giremez","Taşıt Trafiğine Kapalı Yol","Sola Dönüş Yasak","Sağa Dönüş Yasak"
         ,"Öndeki Taşıtı Geçmek Yasak","Hız Sınırı: 70","Yayalar ve Bisikletliler Tarafından Kullanılan Yol","İleri Mecburi Yön"
         ,"Sağa Mecburi Yön","İleri ve Sağa Mecburi Yön","Ada Etrafında Dönünüz","Mecburi Bisiklet Yolu","Yayalar ve Bisikletliler İçin Ayrı Ayrı Kullanılabilen Yol"
         ,"Park Etmek Yasaktır","Duraklamak ve Park Etmek Yasaktır","42:Unknown Sign","43:Unknown Sign","Önceliği Olan Yön"
         ,"Park Yeri","Engelli Park Yeri","Araç Park Yeri","Büyük Araç Park Yeri","Otobüs Park Yeri","50: Unknown Sign"
         ,"Kişisel Yerleşke","Kişisel Yerleşkeden Çıkış","53: Unknown Sign","Yol Bitimi","Kazmak Yasaktır","Yaya Geçidi"
         ,"Bisikletli Geçidi","Park Yeri Girişi","Mecburi Kasisli Yol","60: Unknown Sign","61: Unknown Sign"]

#tkinter app
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)
model = load_model('../savedmodel/my_model.h5')
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

    testimage = load_img(x,target_size=(64,64))
    testimage = img_to_array(testimage)
    testimage = np.expand_dims(testimage, axis=0)
    prediction = model.predict(testimage)
    #print the prediction
    print(np.argmax(prediction[0]))
    var = StringVar()
    label = Label(root, textvariable=var, relief=RAISED )
    # give the output to screen
    var.set("This photo means: "+str(names[np.argmax(prediction[0])]))
    label.pack()
    
btn = Button(root, text='open image', command=open_img).pack()

root.mainloop()

