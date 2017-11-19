import sklearn.decomposition as skd
import tensorflow as tf

import sonnet as snt


class PCA(snt.AbstractModule):
    def __init__(self, n_features, n_components, pca_params={}, name="PCA"):
        super(PCA, self).__init__(name=name)
        self._pca = skd.PCA()
        with self._enter_variable_scope():
            self._mean = tf.get_variable("mean", shape=[n_features])
            self._T = tf.get_variable("T", shape=[n_features, n_components])

    def _build(self, X):
        centered_X = X - self._mean
        # dot product
        X_transformed = tf.matmul(
            centered_X,
            self._T, )
        return X_transformed

    def update(self, session, data):
        transformed_data = self._pca.fit_transform(data)
        session.run([
            self._mean.assign(self._pca.mean_),
            self._T.assign(self._pca.components_.T)
        ])
        return transformed_data
