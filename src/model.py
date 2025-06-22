import tensorflow as tf
from config.config import IMAGE_SIZE, NUM_CLASSES

def build_model():
    # 1)Create input tensor
    input_tensor = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    
    # 2)Load VGG16 base model
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        input_tensor=input_tensor,
        include_top=False
    )
    
    # 3)Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # 4)Build custom head
    x = base_model.output
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    # 5)Create final model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=outputs)
    
    # 6)Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model