import tensorflow as tf

class ComplexSpectralModel:
    def __init__(self, tile_indices=None):
        # Define how to split the 512-point spectrum into tiles.
        # Default: six tiles as in your previous model.
            # CNN kernel slicing indices
        if tile_indices is None:
            self.tile_indices = [(0, 127), (128, 159), (160, 191), (192, 255), (256, 287), (288, 512)]
        else:
            self.tile_indices = tile_indices

    def resnet1d_block(self, x, filters, kernel_size, stride=1, downsample=False):
        """
        A basic 1D residual block.
        """
        shortcut = x
        # First conv
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # Second conv
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if downsample:
            shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    def build_resnet1d_tile_model(self, tile_length, tile_number):
        """
        Build a ResNet1D model (adapted from ResNet-18) for one tile.
        The model takes an input of shape (tile_length, 1) and outputs a fixed-length feature vector.
        """
        inputs = tf.keras.Input(shape=(tile_length, 1))
        # Initial convolution and max pooling
        x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        
        # Group 1: Two residual blocks with 64 filters
        x = self.resnet1d_block(x, filters=64, kernel_size=3, stride=1, downsample=False)
        x = self.resnet1d_block(x, filters=64, kernel_size=3, stride=1, downsample=False)
        
        # Group 2: Two residual blocks with 128 filters, first block downsampling
        x = self.resnet1d_block(x, filters=128, kernel_size=3, stride=2, downsample=True)
        x = self.resnet1d_block(x, filters=128, kernel_size=3, stride=1, downsample=False)
        
        # Group 3: Two residual blocks with 256 filters, first block downsampling
        x = self.resnet1d_block(x, filters=256, kernel_size=3, stride=2, downsample=True)
        x = self.resnet1d_block(x, filters=256, kernel_size=3, stride=1, downsample=False)
        
        # Group 4: Two residual blocks with 512 filters, first block downsampling
        x = self.resnet1d_block(x, filters=512, kernel_size=3, stride=2, downsample=True)
        x = self.resnet1d_block(x, filters=512, kernel_size=3, stride=1, downsample=False)
        
        # Global average pooling to create a fixed-length vector
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Flatten()(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"resnet1d_tile_model_{tile_length}_{tile_number}")

    def build_main_model(self, dropout_rate1 = 0.0, dropout_rate2 = 0.0, dense_units=128):
        """
        Build the main model:
          - Splits the 512-point spectrum into tiles.
          - For each tile, applies a ResNet1D-18 model.
          - Concatenates the resulting features.
          - Passes them through two dense layers.
          - Outputs a two-class softmax (good vs. bad spectrum).
        """
        inp = tf.keras.Input(shape=(512,), name="input_spectrum")
        # Expand dims so each spectrum becomes (512, 1)
        x = tf.expand_dims(inp, axis=-1)
        
        tile_outputs = []
        # Process each tile with its own ResNet1D model
        for i, (start, end) in enumerate(self.tile_indices):
            tile = x[:, start:end, :]  # shape: (batch, tile_length, 1)
            tile_length = end - start
            tile_number = i 
            tile_model = self.build_resnet1d_tile_model(tile_length, tile_number)
            tile_feat = tile_model(tile)
            tile_outputs.append(tile_feat)
        
        # Concatenate features from all tiles
        concat = tf.keras.layers.Concatenate(name="concat_tiles")(tile_outputs)
        
        # Two dense layers followed by the final output layer
        x1 = tf.keras.layers.Dense(dense_units, activation="relu", name="dense1")(concat)
        x1 = tf.keras.layers.Dropout(dropout_rate1, name="dropout1")(x1)  # Add this
        x2 = tf.keras.layers.Dense(dense_units, activation="relu", name="dense2")(x1)
        x2 = tf.keras.layers.Dropout(dropout_rate2, name="dropout2")(x2)  # Add this
        output = tf.keras.layers.Dense(2, activation="softmax", name="output")(x2)
        
        return tf.keras.Model(inputs=inp, outputs=output, name="ComplexSpectralModel")
