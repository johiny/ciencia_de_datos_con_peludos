import gradio as gr
import tensorflow as tf
import numpy as np

title = "Cual es el estado de animo de tu perro?"
description = """
<center >
<h2>Sube una foto de el y descubrelo!</h2>
<img src="https://i.ibb.co/b2SpXk4/Captura-de-pantalla-2023-03-22-122723.png" width=200px style="border-radius:50%;">
</center>
"""
model = tf.keras.models.load_model('./dogs_emotions.h5')
class_names = ['Enojado', 'Feliz', 'Relajado', 'Triste']

def inference(imagen):
    #carga del modelo
    
    #escala la imagen a la resolucion soportada
    imagen = tf.keras.preprocessing.image.smart_resize(
        imagen, (32, 32)
    )

    # expande el array para que este en una sola dimension
    img_array = tf.expand_dims(imagen, 0)

    # defino las diferentes clases y se hacen las predicciones
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    respuesta = "Tu Perro esta {} con un {:.2f}% de seguridad.".format(class_names[np.argmax(score)], 100 * np.max(score))
    return respuesta
 
iface = gr.Interface(fn=inference, inputs=gr.Image(label="Imagen"), outputs=gr.Text(label="Predicción de estado de ánimo"),  title=title,
    description=description)
iface.launch()