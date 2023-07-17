import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import librosa 
import librosa.display
from IPython.display import Audio
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import sounddevice as sd
import wavio
from PIL import Image


def noise(data):
    noise_amp = 0.04*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data, speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data, speed_factor)



#sample_rate = 22050

def extract_features(data):
    
    result = np.array([])
    
    #mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=42) #42 mfcc so we get frames of ~60 ms
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
     
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast') 
    
    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    #noised
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    #stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data)
    result = np.vstack((result, res3))
    
    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))
    
    #pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data)
    result = np.vstack((result, res5)) 
    
    #speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data)
    result = np.vstack((result, res6))
    
    #speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data)
    result = np.vstack((result, res7))
    
    return result




def gender_feature(file_name, **kwargs):
    
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result



with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Loading Gender model
with open("gender_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_gender_model = model_from_json(loaded_model_json)

# Load the model weights from HDF5 file
loaded_gender_model.load_weights("gender_model.h5")
# Compile the loaded model
loaded_gender_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Gender Model loaded successfully.")


# Loading Female Model
with open("female_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_female_model = model_from_json(loaded_model_json)

# Load model weights from HDF5
loaded_female_model.load_weights("female_model_weights.hdf5")
loaded_female_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Female Model loaded successfully!")


# LLoading Male Model
with open("male_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_male_model = model_from_json(loaded_model_json)

# Load model weights from HDF5
loaded_male_model.load_weights("male_model_weights.hdf5")
loaded_male_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Male Model loaded successfully!")



# file = "f0001_us_f0001_00010.wav"
def predict(file):
    features = gender_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = loaded_gender_model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    # # show the result!
    # print(f"Probabilities::: Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
    # print("Result:", gender)
    feat = get_features(file)
    f = feat[1]
    f = np.reshape(f, (1, -1))
    f_scaled = scaler.transform(f)
    if gender == "female":
        prediction = loaded_female_model.predict(f_scaled)
    else:
        prediction = loaded_male_model.predict(f_scaled)
    predicted_label = encoder.inverse_transform(prediction)
    
    emotion = str(predicted_label[0][0])
    return (gender,male_prob,female_prob,emotion)



    



def create_waveshow(data, sr, e):
    fig, ax = plt.subplots(figsize=(10, 3))
    # ax.set_title(f'waveshow for audio with {e} emotion', size=15)
    librosa.display.waveshow(data, sr=sr, ax=ax)
    st.pyplot(fig)

def create_spectrogram(data, sr, e):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig, ax = plt.subplots(figsize=(12, 3))
    # plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    image = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax)   
    fig.colorbar(mappable=image, ax=ax)
    st.pyplot()

def record_audio():
    # Set the sample rate and duration of the recording
    sample_rate = 44100
    duration = 5  # seconds

    # Record audio from the default microphone for the specified duration
    recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)

    # Wait for the recording to complete
    sd.wait()

    # Save the recorded audio to a .wav file
    wav_filename = "recording.wav"
    wavio.write(wav_filename, recording, sample_rate, sampwidth=2)

    # Display a confirmation message
    st.write(f"Audio recorded and saved to {wav_filename}.")



def main():
    st.set_page_config(layout="wide")
    st.title("Speech-Based gender and emotion classifier")
    with st.sidebar:
        st.subheader("Upload File")
        file = st.file_uploader("Choose a WAV file", type="wav")
        st.title("Record Audio")
        if file:
                path = file.name

        # Add a button to trigger the recording
        if st.button("Record"):
            record_audio()
            file = 'recording.wav'
            path = 'recording.wav'

    if file:
        gender,male,female,emotion = predict(path)
        data, sampling_rate = librosa.load(path)
        audio_file = open(path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        gender_class = ['male','female']
        probability = [male,female]

        with st.container():
            st.title(f'Waveform for audio with {emotion} emotion')
            create_waveshow(data, sampling_rate, emotion)
             
        with st.container():
            st.title(f'Spectogram for audio with {emotion} emotion')
            create_spectrogram(data, sampling_rate, emotion)
        

        with st.container():
            col2_left, col2_right = st.columns(2)
            with col2_left:
                colors = ['blue','pink']
                df = pd.DataFrame({'gender_class': gender_class, 'probability': probability})
                st.subheader("Predicted Gender")
                st.success(gender + " " +"with confidence" + " " + str(round(probability[gender_class.index(gender)], 2)))
                fig = alt.Chart(df).mark_bar().encode(x='gender_class', y='probability',color=alt.Color('gender_class', scale=alt.Scale(domain=gender_class, range=colors)))
                st.altair_chart(fig, use_container_width=True)
            with col2_right:
                st.subheader("Final Output")
                gender = gender.capitalize()
                emotion = emotion.capitalize()
                image_path = gender+'_'+emotion+'.PNG'
                image = Image.open(image_path)
                # Display the image
                st.image(image, use_column_width=True)





            
            
                

if __name__ == '__main__':
    main()