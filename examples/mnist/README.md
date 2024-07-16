# MNIST: Multinomial classification 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

```python
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from arnold.layers.polynomial.orthogonal import Hermite

# Define a KAN using Hermite basis polynomials
hermite_kan = tfkl.Sequential([
        tfkl.Reshape(target_shape=(784, )),
        tfkl.Rescaling(scale=1./127.5, offset=-1),
        Hermite(input_dim=784, output_dim=32, degree=2),
        tfkl.LayerNormalization(),
        Hermite(input_dim=32, output_dim=16, degree=3),
        tfkl.LayerNormalization(),
        Hermite(input_dim=16, output_dim=10, degree=2),
        tfkl.Softmax()
    ],
    name="hermite_kan" 
)

# Build, compile and train as usual
hermite_kan.build((None, 28, 28, 1))
hermite_kan.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
hermite_kan.fit(
    mnist_train,
    epochs=EPOCHS, 
    shuffle=True,
    verbose=1
)
```

