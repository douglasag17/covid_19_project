import pandas as pd
import sklearn as sk
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
import geopandas as gpd
import numpy as np
from abc import ABCMeta, abstractmethod


class clustering_interface(metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data
        self.embeddings = data.values
        self.clusters = {}

    @abstractmethod
    def reduce_dimensionality(self):
        pass

    @abstractmethod
    def fit_clusters(self, number_clusters):
        pass
"""
    @property
    def data(self):
        return self.data

    @property
    def embeddings(self):
        return self.embeddings

    @property
    def clusters(self):
        return self.clusters

    @data.setter
    def data(self, data):
        self.data = data

    @embeddings.setter
    def data(self, embeddings):
        self.embeddings = embeddings
"""



class pca_kmeans(clustering_interface):
    def __init__(self, data):
        super().__init__(data)

    def reduce_dimensionality(self):
        for i in range(len(normalized_df.columns)):
            pca = PCA(i)
            pca.fit(self.data)
            if sum(pca.explained_variance_ratio_) > 0.99:
                break
            self.embeddings = pca.transform(self.data)

    def fit_clusters(self, number_clusters):
        kmeans = KMeans(number_clusters,random_state=42)
        kmeans.fit(self.embeddings)
        clusters_kmeans = kmeans.predict(self.embeddings)
        self.clusters = { index:clusters_kmeans[i] for i,\
                             index in enumerate(self.data.indnex) }

class autoencoders_kmeans(clustering_interface):
    def __init__(self, data):
        super().__init__(data)

    def reduce_dimensionality(self,original_dimension):
        encoder = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='selu', input_shape=[original_dimension]),
        tf.keras.layers.Dense(3, activation= 'selu')
        ])
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='selu'),
            tf.keras.layers.Dense(original_dimension, activation='sigmoid')
        ])

        stacked_ae = tf.keras.models.Sequential([encoder, decoder])
        stacked_ae.compile(loss = 'mse', optimizer=tf.keras.optimizers.SGD(lr=0.1))
        history = stacked_ae.fit(self.data, self.data, epochs=20)
        codings = encoder.predict(self.data)
        self.embeddings = codings

    def fit_clusters(self, number_clusters):
        kmeans = KMeans(number_clusters,random_state=42)
        kmeans.fit(self.embeddings)
        clusters_kmeans = kmeans.predict(self.embeddings)
        self.clusters = { index:clusters_kmeans[i] for i,\
                             index in enumerate(self.data.index) }

