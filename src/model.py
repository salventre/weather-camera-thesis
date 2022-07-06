from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

INPUT_SHAPE = (160,32,3) #to verify
N_CLASSES = 4 #dry, wet, snow, fog
N_LAYERS_TO_TRAIN = 10 #to modify

def build_model(freeze:bool=None)->Model:
    base_model = MobileNetV2(INPUT_SHAPE, include_top=False, weights='imagenet') # 154 layers
    if freeze is not None:
        if freeze: base_model.trainable = False
        else:
            for layer in base_model.layers:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.2)(x)
    model = Dense(N_CLASSES, 'softmax')(x)
    final_model = Model(inputs=base_model.input, outputs=model)
    #print(final_model.summary())
    return final_model