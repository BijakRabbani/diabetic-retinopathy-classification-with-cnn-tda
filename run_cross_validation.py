#%%
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Preprocessing import PreprocessingPipeline
from Model import create_model_instance
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from Callbacks import Metrics, step_decay
import datetime as dt


if __name__ == '__main__':
    #%% Data parameters
    SEED = 10
    image_path = 'data/image/'
    label_path = 'data/train.csv'

    data_label = pd.read_csv(label_path, index_col='id_code')
    files = [i.split('.')[0] for i in os.listdir(image_path)]
    files = [i for i in files if i != '']
    data_label = data_label.loc[files]

    _, _, y_dev, y_val = train_test_split(
        data_label, data_label, test_size=0.2, random_state=SEED
        )
    #%% Build model
    params = {
        'batch_size': 20,
        'image_size': (128, 128),
        'topo_dim': 200,
        'beta': 0.1,
        'topo_channel': 'r',
        'resnet_depth': 9*2+2,
        'dropout_rate': 0.5,
        'fully_connected_node': 1000,
        'label_smoothing': 0.1,
        'epochs': 100
    }

    img_gen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20, 
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode='constant',
        cval=0,
        )

    img_gen_val = ImageDataGenerator(
        rescale=1./255, 
        )

    model = create_model_instance(params, model_name='cnn_betti_curve')
    #%%
    skf = StratifiedKFold(n_splits=2)
    cv_result = {}
    for fold, (train_id, val_id) in enumerate(skf.split(y_dev, y_dev)):
        
        # Define preprocessing pipeline
        train_gen = PreprocessingPipeline(
            y_dev.iloc[train_id], 
            params['batch_size'], 
            img_gen, 
            image_path,
            topo_channel=params['topo_channel'],
            )
        val_gen = PreprocessingPipeline(
            y_dev.iloc[val_id], 
            params['batch_size'], 
            img_gen_val, 
            image_path,
            topo_channel=params['topo_channel'],
            )

        train_step = int(np.ceil(train_gen.samples_size / params['batch_size']))
        val_step = int(np.ceil(val_gen.samples_size / params['batch_size']))

        # Define callbacks
        lrate = LearningRateScheduler(step_decay)
        kappa_metrics = Metrics(
            model, 
            val_gen, 
            val_step, 
            params['batch_size']
            )
        callbacks_list = [kappa_metrics, lrate]

        #% Training
        history = model.fit(train_gen, 
                            steps_per_epoch = train_step, 
                            epochs = params['epochs'],
                            validation_data=val_gen,
                            validation_steps=val_step,
                            callbacks = callbacks_list
                            )

        history.history['kappa'] = kappa_metrics.val_kappas
        cv_result[fold] = pd.DataFrame(history.history)

    # Collect results
    metrics = pd.concat(cv_result.values(), keys=cv_result.keys(), axis=1)
    mean = (
        metrics
        .groupby(level=1, axis=1)
        .mean().T
        .assign(k='mean')
        .set_index('k',append=True)
        .swaplevel(0,1).T
    )
    std = (
        metrics
        .groupby(level=1, axis=1)
        .std().T
        .assign(k='std')
        .set_index('k',append=True)
        .swaplevel(0,1).T
    )
    metrics = pd.concat([metrics, mean, std], axis=1)

    #%% Store results
    result_path = 'result/' + dt.datetime.today().strftime('%Y%m%d_%H%M%S')
    os.mkdir(result_path)
    metrics.to_csv(result_path+'/metrics.csv')
    pd.Series(params).to_csv(result_path+'/parameters.csv')