import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class EmotionPredictor():
    def __init__(self, path = "models/efficientB0_Affect.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def process(self, image):
        image = tf.image.resize(image, (224,224), preserve_aspect_ratio=True)
        image = np.reshape(image, (1, 224, 224, 3))
        return image
    
    def get_label(self, i):
        mapper = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Sad',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger',
        7: 'Contempt'
             }
        
        return mapper[i]
    
    def predict(self, x):
        x = self.process(x)
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[1]['index'])
        idx = np.argmax(output_data[0])
        
        return self.get_label(idx), round(output_data[0][idx]*100, 2)

def plot_learning_curves(history, name, metric = 'sparse_categorical_accuracy', save = None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title(f'Model Loss - {name}')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend(['train', 'validation'], loc='upper left')
    
    ax[1].plot(history.history[metric])
    ax[1].plot(history.history[f'val_{metric}'])
    ax[1].set_title(f'Model Accuracy - {name}')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(['train', 'validation'], loc='upper left')
    fig.suptitle(f"Learning Curves for {name}")
    fig.tight_layout()
    if save is not None:
        plt.savefig(f'images/{save}.png')
    plt.show()

def build_data(path = 'train', rebuild = False):
    if not rebuild and os.path.exists(f'data/affect/processed_{path}.csv'):
        return pd.read_csv(f'data/affect/processed_{path}.csv')
    
    image_path = f'data/affect/{path}_set/images/'
    anno_path = f'data/affect/{path}_set/annotations/'
    image_names = os.listdir(image_path)
 
    data = {
        'valence': [],
        'arousal': [],
        'expression': [],
        'emotion': [],
        'image_path': []
    } 
    mapper = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Sad',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger',
        7: 'Contempt'
             }
    
    for i in image_names:
        img_id = i.split('.')[0]
        data['valence'].append(np.load(anno_path + f'{img_id}_val.npy'))
        data['arousal'].append(np.load(anno_path + f'{img_id}_aro.npy'))
        data['expression'].append(np.int(np.load(anno_path + f'{img_id}_exp.npy')))
        data['emotion'].append(mapper[data['expression'][-1]])
        data['image_path'].append(image_path + i)
    
    
    
    df = pd.DataFrame(data)
    df.to_csv(f'data/affect/processed_{path}.csv')
    
    return df