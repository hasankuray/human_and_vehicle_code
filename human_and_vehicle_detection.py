import cv2
import numpy as np
import sqlite3
import datetime
import os

# veritabanımıza bağlantı sağlıyoruz.
con = sqlite3.connect("kayitlar.db")
cursor = con.cursor()

# veritabanında blob veri türünde fotoğraf saklamak istediğimiz için fotoğrafları binary formatına çeviriyoruz.
def foto_binary(filename):
    
    with open(filename, 'rb') as file:
        blob_data = file.read()
    return blob_data

# veritabanına kaydetme sorgusu ve veritabanına kaydedeceğimiz bilgileri içeren fonksiyon
def insert_blob(class_name,photo,time):
    
    sqlite_query = "INSERT INTO kayit_tablosu (class_name,photo,date) VALUES (?,?,?)"
    
    binary_photo = foto_binary(photo)
    data_tuple = (class_name, binary_photo, time)
    
    cursor.execute(sqlite_query,data_tuple)
    con.commit()

cap = cv2.VideoCapture("araba_video4.mp4")
i = 0


#%%

while True: 
    ret, frame = cap.read()
    height,width,pix = frame.shape
    
    # resmi modele verebilmek için 4 boytulu hale getirip üzerinde aşağıdaki işlemleri yapıyoruz.
    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB = True,crop = False)
    
    labels = ["Human","Vehicle"] 
     
    # yoloda eğittiğimiz weights dosyamız ve eğitimde kullandığımız cfg dosyamızı burada kullanıp modelimizi oluşturuyoruz.
    model = cv2.dnn.readNetFromDarknet("human_and_vehicle.cfg","human_and_vehicle_4.weights")    
    
    # modelimizin çıkış katmanlarını düzenliyoruz.
    layers = model.getLayerNames()
    output_layers =  [layers[int(layer)-1] for layer in model.getUnconnectedOutLayers()]
    
    # az önce blob formatına çevirdiğimiz resmimizi modelimize tahmin etmesi için girdi olarak veriyoruz.
    model.setInput(frame_blob)
    
    # model tahmin işlemini gerçekleştriyor.
    detection_layers = model.forward(output_layers)
    
    # non max suppression 1
    confidence_list = []
    box_list = []
    ids_list = []
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            # sınıflarımıza ait tahmin sonuçlarımızı burada alıp en yüksek sınıfa ait değerleri tutuyoruz.
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            # tahminimiz sonucunda oluşan değer aşağıdaki değerimizden büyük ise nesnemizi tespit etmiş oluyoruz.
            if confidence > 0.35:
                
                # bounding box kenarlarını belirleme
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([width,height,width,height])      #  ilk 5 değer bounding box ile ilgili
                box_center_x , box_center_y , box_width , box_height = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                # non max suppression 2
                confidence_list.append(float(confidence))
                box_list.append([start_x,start_y,int(box_width),int(box_height)])
                ids_list.append(predicted_id)

    # non max suppression 3
    max_ids = cv2.dnn.NMSBoxes(box_list,confidence_list,0.12,0.1)

    for max_id in max_ids:
        max_class_id = max_id[0]
        box = box_list[max_class_id]
        
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
        
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        
        confidence = confidence_list[max_class_id]
        
        
        end_x = start_x + box_width
        end_y = start_y + box_height
                
        
        label = "{}: {:.2f}%".format(labels[predicted_id],confidence*100)
        print("predicted object {}".format(label))
           
        # bounding box ve labelimizi frame üzerine çizdiriyoruz.
        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),(0,0,255),2)
        cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        
        # nesnemiz bulunduğu zaman sadece nesnemizin fotoğrafını alıyoruz.
        nesne = frame[start_y:end_y,start_x:end_x]
        
        #kayıt işlemi
        i += 1
        resim_yolu = "resimler/" + str(i) + ".jpg"
        cv2.imwrite(resim_yolu,nesne)
        
        #zamanı öğrenme
        time = datetime.datetime.now()
        
        #veritabanı kayıt
        if(predicted_id == 0):
            insert_blob("Human", resim_yolu,time)
        elif(predicted_id == 1):
            insert_blob("Vehicle",resim_yolu,time)
                
        os.remove(resim_yolu)
        
    cv2.imshow("Window",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cursor.close()
con.close()
cap.release()
cv2.destroyAllWindows()
