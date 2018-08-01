from keras import Input, Model
from keras.layers import BatchNormalization, GaussianNoise, Flatten, Embedding, Concatenate
import pandas as pd


class EmbeddingNN:

    def __init__(self, quantitative_layers, concat_layers,
                 loss, optimizer='adam'):
        self.optimizer = optimizer
        self.loss = loss
        self.concat_layers = concat_layers
        self.quantitative_layers = quantitative_layers
        self.fithist = []
        self._model: Model = None

    def _split_matrix(self, m, catcols, quacols):
        return {'Quant': m[:, quacols],
                **{'Categ_' + str(catcol): m[:, catcol] for catcol in catcols}}

    def _embeds_nn(self, m, catcols, embed_size):
        if not isinstance(embed_size, (list, tuple)):
            value_embed = embed_size
            embed_size = [value_embed for _ in range(len(catcols))]
        for i, catcol in enumerate(catcols):
            categ_input = Input(shape=(1,), name='Categ_' + str(catcol))
            categ = Embedding(input_dim=m[:, catcol].max() + 1,
                              output_dim=embed_size[i])(categ_input)
            categ = Flatten()(categ)
            yield categ_input, categ

    def _quant_nn(self, n_quacols):
        quant_input = Input(name='Quant', shape=(n_quacols,))
        quant = quant_input
        for i in range(len(self.quantitative_layers)):
            quant = self.quantitative_layers[i](quant)
        return quant_input, quant

    def _concat_nn(self, qualayer, *embedlayers):
        concat = Concatenate(axis=1)([*embedlayers, qualayer])
        for i in range(len(self.concat_layers)):
            concat = self.concat_layers[i](concat)
        return concat

    def fit(self, X, y, catcols=None, quacols=None, embed_size=5, **kwargs):
        X, y, catcols, quacols = self._arrange_input(X, y, catcols, quacols)
        categ_inputs, embeddings = zip(*list(self._embeds_nn(X, catcols, embed_size)))
        qua_input, qualayer = self._quant_nn(len(quacols))
        output_layer = self._concat_nn(qualayer, *embeddings)
        X = self._split_matrix(X, catcols, quacols)
        model = Model(inputs=[qua_input, *categ_inputs], outputs=output_layer)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        hist = model.fit(X, y, **kwargs)
        self.fithist.append(hist)
        self._catcols = catcols
        self._quacols = quacols
        self._model: Model = model
        return hist

    def _arrange_input(self, X, y, catcols, quacols):
        assert not (catcols is None and quacols is None)
        if isinstance(X, pd.DataFrame):
            if isinstance(y, str):
                y_label = y
                y = X[y_label].values
                X = X.drop(columns=y_label)
            if isinstance(catcols, list) and isinstance(catcols[0], str):
                catcols = [i for i, col in X.columns for catcol in catcols if col == catcol]
            if isinstance(quacols, list) and isinstance(quacols[0], str):
                quacols = [i for i, col in X.columns for quacol in quacols if col == quacol]
            X = X.get_values()
        if catcols is None:
            catcols = [i for i in (set(range(X.shape[1])) - set(quacols))]
        if quacols is None:
            quacols = [i for i in (set(range(X.shape[1])) - set(catcols))]
        return X, y, catcols, quacols

    def predict(self, X, batch_size=None, verbose=0, steps=None):
        assert hasattr(self, '_model') and self._model, 'Model not trained'
        X = self._split_matrix(X, self._catcols, self._quacols)
        return self._model.predict(X, batch_size, verbose, steps)
