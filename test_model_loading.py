import tensorflow as tf

# Load your SavedModel
model = tf.keras.models.load_model("saved models/deep_learning_classifier")

# Save the model in .h5 format
model.save("saved models/deep_learning_classifier.h5")

