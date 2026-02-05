import os
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.server.strategy import FedAvg
from fl_ids.strategy import CustomFedAvg

# Importamos las utilidades de tu proyecto
import fl_ids.utils.model_loader as model_loader
import fl_ids.utils.data_loader as data_loader

# Inicialización de la ServerApp
app = ServerApp()

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # This is the exact same dataset as the one downloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    _, _, X_test, y_test = data_loader.get_data()
    global_model =  model_loader.get_model(X_test.shape[1:])

    global_model.set_weights(arrays.to_numpy_ndarrays())
    loss, accuracy = global_model.evaluate(X_test, y_test, verbose=0)
    return MetricRecord({"accuracy": accuracy, "loss": loss})

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval = context.run_config["fraction-evaluate"]
    
    X_train, _, _, _ = data_loader.get_data()
    global_model =  model_loader.get_model(X_train.shape[1:])
    
    # Convertimos los pesos a ArrayRecord para el transporte de Flower
    initial_arrays = ArrayRecord(global_model.get_weights())

    # strategy = CustomFedAvg(
    #     fraction_train=fraction_train,
    #     fraction_evaluate=fraction_eval,
    #     min_train_nodes=context.run_config.get("min-fit-clients", 3),
    #     min_evaluate_nodes=context.run_config.get("min-evaluate-clients", 3),
    #     min_available_nodes=context.run_config.get("min-available-clients", 3),
    # )
    
    # Initialize FedAvg strategy
    strategy = CustomFedAvg(
        fraction_train=fraction_train, fraction_evaluate=fraction_eval
    )
    
    # Define directory for results and save config
    save_path, run_dir = model_loader.create_run_dir(config=context.run_config)
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": 0.001}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )


    print("\n Completed training. Saving the final model...")
    global_model.set_weights(result.arrays.to_numpy_ndarrays())
    global_model.save("final_model.keras")
        