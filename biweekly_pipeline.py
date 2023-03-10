import os
import modal

stub = modal.Stub()

training_image = modal.Image.debian_slim().pip_install(["hopsworks", "tensorflow", "tensorflow-hub", "scikit-learn", "tensorflow-text"])

@stub.function(image=training_image, schedule=modal.Period(days=3), secret= modal.Secret.from_name("hopswork-api-key"), timeout=5400, retries=2)
def train_model():
    """Train stance detection model and store trained model on Hopsworks Model Registry."""
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text
    from sklearn.model_selection import train_test_split

    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    print("INFO: Connecting to Hopsworks and reading training data from Feature Store")
    project = hopsworks.login()
    feature_store = project.get_feature_store()
    article_feature_group = feature_store.get_feature_group(name="training_data_stance", version=1)
    data = article_feature_group.read()
    data = data.rename(columns={'title': 'title', 'predicted_topic': 'subjects', 'stance': 'stance'})
    # as somehow Hopsworks ended up with NaNs at random rows, filter these rows
    data = data[data['stance'].notnull()]
    # convert from 1=neg, 2=neu, 3=pro to 0=neg, 1=neu, 2=pro
    data['stance'] = data['stance']-1
    data['stance'] = data['stance'].astype('category')
    
    X_train, X_test, y_train, y_test = train_test_split(data[['title', 'subjects']], data['stance'], test_size=0.33)

    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"
    hidden_layer_sizes = [300, 100]
    alpha_leaky_relu = 0.01
    dropout = 0
    number_of_output_classes = 3

    def build_classifier_model():
        # Handle categorical labels
        # encoded_stance = get_category_encoding_layer(name='categorical', dataset=data_train, dtype='string')

        # BERT embeddings
        preprocessor = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')

        title = tf.keras.layers.Input(shape=(), dtype=tf.string, name='title')
        titles_preprocessed = preprocessor(title)

        subjects = tf.keras.layers.Input(shape=(), dtype=tf.string, name='subjects')
        subjects_preprocessed = preprocessor(subjects)

        # pooled_output gives the embedding per input sequence, alternatively sequence_output would give it per input token
        titles_embedded = encoder(titles_preprocessed)["pooled_output"]
        subjects_embedded = encoder(subjects_preprocessed)["pooled_output"]

        # Concat features
        # titles_embedded = tf.keras.layers.Flatten()(titles_embedded)
        # subjects_embedded = tf.keras.layers.Flatten()(subjects_embedded)
        concatenated_features = tf.keras.layers.Concatenate()([titles_embedded, subjects_embedded])
        out = concatenated_features

        # Neural network hidden layers
        for number_of_hidden_units in hidden_layer_sizes:
            out = tf.keras.layers.Dense(number_of_hidden_units, activation=tf.keras.layers.LeakyReLU(alpha=alpha_leaky_relu))(out)
            if dropout > 0:
                out = tf.keras.layers.Dropout(dropout)(out)

        # Output
        out = tf.keras.layers.Dense(number_of_output_classes, activation=tf.nn.softmax, name='prediction')(out)

        return tf.keras.Model(
            inputs = [title, subjects],
            outputs = out
        )

    model = build_classifier_model()
    # model.summary()
    # tf.keras.utils.plot_model(model)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

    epochs=5
    history = model.fit(
        x=[
            X_train['title'],
            X_train['subjects']
        ],
        y=y_train,
        epochs=epochs,
        validation_data=([X_test['title'], X_test['subjects']], y_test)
    )

    # Thanks to Jim Dowling for inspiration for the code below
    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    print("INFO: Getting Hopsworks Model registry for storing trained model")
    mr = project.get_model_registry()
    
    # The contents of the 'stance_model' directory will be saved to the model registry. Creating the dir first.
    model_dir="stance_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save Keras model in SavedModel format
    model.save(model_dir)
 
    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    stance_model = mr.python.create_model(
        name="stance_model", 
        metrics={
            "accuracy": history.history['accuracy'][-1], 
            "validation_accuracy": history.history['val_accuracy'][-1],
            },
        model_schema=model_schema,
        description="Articles stance predictor"
    )
    
    # Upload the model to the model registry, including all files in 'model_dir'
    stance_model.save(model_dir)


if __name__ == "__main__":
    # Programatic deployment of biweekly schedule
    stub.deploy("biweekly_pipeline")
    with stub.run():
        try:
            train_model()
        except modal.exception.TimeoutError:
            print("Timeout for model training. Try manual rerun or increase retry policy from current setting 2")
