import numpy as np
import librosa
import os
import pandas as pd
import tensorflow as tf
import shutil

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

def load_model():
    model = tf.keras.models.load_model('/home/ammar/Desktop/VectraCom/mustanad_jawab/saved_models/model_(3)_95.hdf5')
    return model

if __name__ == "__main__":
    acc=0
    rej=0
    err=0
    labels = list()
    filename = list()
    audio_path = "assets/Mufti_Folder/MUFTI_QUESTIONS"
    files = os.listdir(audio_path)
    model = load_model()
    os.system("clear")
    for file in files:
        try:    
            filen = os.path.join(audio_path,file)
            data = features_extractor(filen)
            prediction_feature = data.reshape(1,-1)
            class_prob = model.predict(prediction_feature)
            print(class_prob)
            predicted_label = 0
            if class_prob[0][0] > 0.80 and class_prob[0][0] > class_prob[0][1]:
                predicted_label = 0
            else:
                predicted_label = 1
                
            if predicted_label == 0:
                print("Accepted")
                dest = os.path.join("/home/ammar/Desktop/VectraCom/mustanad_jawab/assets/big_test_output/accepted",file)
                shutil.copy(filen,dest)
                acc+=1
                filename.append(file)
                labels.append(0)    #accepted
            else:
                print("Rejected")
                dest = os.path.join("/home/ammar/Desktop/VectraCom/mustanad_jawab/assets/big_test_output/rejected",file)
                shutil.copy(filen,dest)
                filename.append(file)
                rej+=1
                labels.append(1)    #rejected
            print(f"Processed | {file}")
        except:
            print(f"Error with file | {file}")
            err+=1
            
    df = pd.DataFrame(list(zip(filename,labels)),columns=["file","status"])
    df.to_csv("/home/ammar/Desktop/VectraCom/mustanad_jawab/assets/big_test_output/test_log.csv", sep='|', index=False)
    print(f"Error Files: {err}\nTotal Processed Files: {acc+rej}\nAccepted: {acc}\nRejected: {rej}")