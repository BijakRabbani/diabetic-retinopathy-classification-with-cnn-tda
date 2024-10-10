import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn_tda import DiagramSelector, DiagramScaler, Landscape, BettiCurve, PersistenceImage
from sklearn_tda.preprocessing import Clamping
import math
import dionysus as d
from tensorflow.keras.utils import Sequence, to_categorical


class PreprocessingPipeline(Sequence):

    def __init__(self, data_label, batch_size, augmentation, 
                image_folder, image_size=(128, 128), 
                shuffle=True, transform_target=True, topo_channel='r'):
        self.data_label = data_label
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_size = image_size
        self.shuffle = shuffle
        self.topo_channel = topo_channel
        self.image_folder = image_folder
        self.on_epoch_end()
        self.transform_target = transform_target
        self.topo_features = {}
        self.samples_size = len(data_label)

    def __len__(self):
        return math.ceil(len(self.data_label) / self.batch_size)

    def __getitem__(self, idx):
        # Prepare image metadata
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = self.data_label.iloc[indexes]       
        batch_data = batch_data.reset_index()
        batch_data['id_code'] = batch_data['id_code'] + '.png'
        y = to_categorical(batch_data['diagnosis'], num_classes=5)
        batch_data['diagnosis'] = batch_data['diagnosis'].astype(str)

        # Image augmentation
        batch_gen = self.augmentation.flow_from_dataframe(
                                            dataframe=batch_data,
                                            directory=self.image_folder,
                                            x_col='id_code',
                                            y_col='diagnosis',
                                            classes=['0','1','2','3','4'],
                                            target_size=self.image_size,
                                            batch_size=self.batch_size,
                                            shuffle=False
                                            )
        image_x, temp_y = batch_gen.next()

        # Extract topological features
        channel_to_id = {
            'r': 0,
            'g': 1,
            'b': 2,
        }
        topo_feature_value = []
        for image_id in range(image_x.shape[0]):
            persistence_diagram = self.get_persistence_diagram(image_x[image_id,:,:,channel_to_id[self.topo_channel]])
            curr_topo_feat = self.betti_curve(persistence_diagram)
            curr_topo_feat_norm = MinMaxScaler().fit_transform(curr_topo_feat.reshape(-1, 1))
            topo_feature_value.append(curr_topo_feat_norm[:,0])
        topo_feature_value = np.array(topo_feature_value)
        
        return ([np.array(image_x),topo_feature_value], np.array(temp_y))
    
    def get_persistence_diagram_points(self, dgm):
        point_list = []
        X = []
        Y = []
        max_death = 0
        for point in dgm:
            if max_death < point.death and point.death != np.inf:
                max_death = point.death

        for point in dgm:
            if point.death == np.inf:
                death = max_death
            else:
                death = point.death
            birth = point.birth
            point_list.append([birth, death])
            X.append((death + birth) / 2)
            Y.append((death - birth))

        if not X:
            X.append(0) 
        if not Y:
            Y.append(0)

        X = np.array(X)
        Y= np.array(Y)
        point_list = np.array(point_list)
        return point_list, X, Y

    def get_persistence_diagram(self, image_matrix):
        f_lower_star = d.fill_freudenthal(image_matrix)
        p = d.homology_persistence(f_lower_star)
        dgms = d.init_diagrams(p, f_lower_star)
        points_0 = self.get_persistence_diagram_points(dgms[0])[0]
        points_1 = self.get_persistence_diagram_points(dgms[1])[0]
        return points_0, points_1

    def betti_curve(self, dgms):
        '''
        Betti numbers for increasing epsilon (Umeda, 2017)
        '''
        result = np.array([])
        for i in range(len(dgms)):
            D = dgms[i]
            diags = [D]
            
            diags = DiagramSelector(use=True, point_type="finite").fit_transform(diags)
            diags = DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())]).fit_transform(diags)
            diags = DiagramScaler(use=True, scalers=[([1], Clamping(limit=.9))]).fit_transform(diags)
            
            res = 100
            BC = BettiCurve(resolution=res)
            bc = BC.fit_transform(diags)    
            result = np.append(result, bc[0])
        return result

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.data_label))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)