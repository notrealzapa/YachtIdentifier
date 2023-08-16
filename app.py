import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("best_model.h5")

st.title("Yacht Classification Web App")
st.markdown("""
Style and class, right at your fingertips. 🛥️
Upload an image of a yacht, and this app will predict its class.
""")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Yacht Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    top_5_indices = predictions[0].argsort()[-5:][::-1]
    top_5_probs = predictions[0][top_5_indices]
    classes = ['A', 'Nord', 'OK', 'Yas', 'a+', 'afterglow', 'al mirqab', 'al said', 'al salamah', 'alegria', 'alfa', 'amaryllis', 'anastasia', 'andrea', 'aphrodite', 'arados', 'aresteas', 'atlantic', 'atom', 'azimut 27 grande', 'azzam', 'baton rouge', 'beatrix', 'belle anna', 'benetti oasis', 'big eagle', 'big fish', 'blue', 'capri 1', 'carinthia vii', 'clio', 'cloud atlas', 'cocoa bean', 'crescent', 'crocus', 'curfew 2', 'curiosity', 'deep blue', 'dilbar', 'dojo', 'dragonfly', 'dreamer', 'dubai', 'eclipse', 'edesia', 'el caran', 'el mahrousa', 'el mirar 2', 'ezgy', 'falco moscata', 'firebird', 'flying fox', 'flying manta', 'force blue', 'formosa', 'four jacks', 'fx', 'g3', 'geco', 'ghost 2', 'grade 1', 'harmony 3', 'here comes the sun', 'ice', 'illusion plus', 'jongert project revolution', 'k2', 'katara', 'khalilah', 'king baby', 'king k', 'lady britt', 'lady e', 'lady lucy', 'latiko', 'liberty', 'lusia m', 'mahana', 'maria', 'mary jean 2', 'masteka 2', 'maveric 1', 'mirage', 'miraggio', 'moka', 'moonlight 2', 'nauta', 'nereus', 'ocean victory', 'octopus', 'odessa', 'odyssea', 'okto', 'oriana', 'ouranos', 'paloma', 'panfeliss', 'paradigm', 'party girl', 'persefoni', 'planet nine', 'polestar', 'prince abdulaziz', 'quantum blue', 'rania', 'raph seven', 'rising son', 'rock 13', 'roma', 'saluzi', 'samurai', 'savorona', 'scheherazade', 'seagull 2', 'seanna', 'seawolf', 'serene', 'seychelle', 'sherakhan', 'sirahmy', 'so nice', 'solaris', 'soundwave', 'spirit', 'splendida', 'starfire', 'startup', 'status quo', 'storm', 'super veloce', 'tango', 'tatoosh', 'tie breaker', 'titania', 'triumph', 'ultra violet 2', 'unbridled', 'vagabond', 'veneta', 'vertige', 'vibrance', 'victorious ', 'virtus xp hull 02', 'wheels', 'x', 'y721', 'yemanja', 'yersin', 'zamboanga']
    top_5_classes = [classes[i] for i in top_5_indices]

    st.subheader("Top 5 Predicted Classes:")
    for i in range(5):
        st.write(f"{i+1}. Class: {top_5_classes[i]}, Probability: {top_5_probs[i]*100:.2f}%")

st.write("---")
st.markdown("""
### Built with ❤️ by [Zhumazhan Balapanov](#)

Find me on:
- [GitHub](https://github.com/notrealzapa)
- [LinkedIn](https://www.linkedin.com/in/zhumazhan-balapanov-679a23218/)
""")

