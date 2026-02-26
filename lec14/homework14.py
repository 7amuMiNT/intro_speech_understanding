import gtts, speech_recognition, librosa, soundfile

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)

def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognizer = sr.Recognizer()
    recognized_texts = []

    for text, lang, root in zip(texts, languages, filenames):
        mp3file = root + ".mp3"
        wavfile = root + ".wav"

        # 1) synthesize MP3
        synthesize(text, lang, mp3file)

        # 2) convert MP3 -> WAV using librosa + soundfile
        y, Fs = librosa.load(mp3file, sr=None, mono=True)
        sf.write(wavfile, y, Fs)

        # 3) recognize WAV
        with sr.AudioFile(wavfile) as source:
            audio = recognizer.record(source)
        recognized = recognizer.recognize_google(audio, language=lang)

        recognized_texts.append(recognized)

    return recognized_texts
