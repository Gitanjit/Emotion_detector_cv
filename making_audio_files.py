
from playsound import playsound
from gtts import gTTS

tts = gTTS("No Faces")
tts.save('nofaces.mp3')

playsound('nofaces.mp3')
