import os
import keras
from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

import fl_ids.utils.data_loader as data_loader
import fl_ids.utils.model_loader as model_loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Init ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    keras.backend.clear_session()

    x_train, y_train, _, _ = data_loader.get_data()

    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    
    model = model_loader.get_model(x_train.shape[1:])
    
    parameters = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(parameters)

    history = model.fit(
        x_train, 
        y_train, 
        epochs=local_epochs, 
        batch_size=batch_size, 
        verbose=0
    )

    metrics = {k: float(v[-1]) for k, v in history.history.items()}
    
    # Construimos el contenido de la respuesta
    # 'arrays' reemplaza al primer elemento de tu tupla (model.get_weights())
    # 'metrics' incluye el diccionario de métricas
    content = RecordDict({
        "arrays": ArrayRecord(model.get_weights()),
        "metrics": MetricRecord(metrics)
    })
    
    # Añadimos metadatos útiles como el número de ejemplos (len(X_train))
    content["metrics"]["num_examples"] = len(x_train)
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Equivalente al método evaluate() del NumPyClient.
    """
    keras.backend.clear_session()

    _, _, x_test, y_test = data_loader.get_data()

    model = model_loader.get_model(x_test.shape[1:])
    parameters = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(parameters)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "num_examples": len(x_test),
    }
    
    content = RecordDict({
        "metrics": MetricRecord(metrics)
    })
    
    return Message(content=content, reply_to=msg)