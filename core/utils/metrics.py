from typing import Dict, List, Tuple, Union

class Metrics():
    def __init__(self, metrics: List[str]):
        metrics.insert(0, "epoch")
        self._metrics: Dict[str, List[Tuple[str, Union[float, int]]]] = {metric: [] for metric in metrics}


    def __getitem__(self, key):
        return self._metrics[key]
    

    def __iter__(self):
        return iter(self._metrics.items())
    

    def __repr__(self):
        metric_string = "Metrics: {\n"
        for key, value in self._metrics.items():
            item_string = f"    {key}: {value}\n"
            metric_string += item_string

        metric_string += "}"
        return metric_string
    

    def get_metrics_dict(self) -> Dict[str, List[Tuple[str, Union[float, int]]]]:
        return self._metrics


    def update(self, epoch, batch_loss, **kwargs) -> None:
        self._metrics["epoch"].append(epoch+1)
        self._metrics["loss"].append(batch_loss)

        for key, value in kwargs.items():
            self._metrics[key].append(value)