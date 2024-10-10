from ImageLayer import resnet_v2
from tensorflow.keras.layers import (
    Input,
    Flatten,
    concatenate,
    Dropout,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model


def create_model_instance(params, model_name):
    # Image layer
    image_input = Input(shape=tuple(list(params['image_size']) + [3]))
    image_layer = resnet_v2(
        input_layer=image_input,
        depth=params['resnet_depth'], 
        num_classes=5
        )

    # TDA layer
    topo_input = Input(shape=params['topo_dim'])
    topo_layer = Flatten()(topo_input)
    topo_layer = Dropout(params['dropout_rate'])(topo_layer)

    # Merge
    merge_layer = concatenate([image_layer, topo_layer], axis=-1)

    # Classification
    FC = Dense(params['fully_connected_node'], activation='relu')(merge_layer)
    FC = Dropout(params['dropout_rate'])(FC)
    FC = Dense(params['fully_connected_node'], activation='relu')(FC)
    FC = Dropout(params['dropout_rate'])(FC)
    prediction = Dense(5, activation='softmax')(FC)

    # Compile
    model = Model(
        inputs=[image_input, topo_input], 
        outputs=[prediction]
        )
    opt = Adam(beta_1=params['beta'])
    loss = CategoricalCrossentropy(
        label_smoothing=params['label_smoothing']
        )
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.model_name = model_name
    return model