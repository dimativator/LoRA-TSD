from . import boolq, multi_nli, openbookqa, piqa, qnli, siqa


DATASETS = {
    "boolq": (boolq.get_train_dataloader, boolq.compute_boolq_accuracy),
    "piqa": (piqa.get_train_dataloader, piqa.compute_piqa_accuracy),
    "siqa": (siqa.get_train_dataloader, siqa.compute_siqa_accuracy),
    "obqa": (openbookqa.get_train_dataloader, openbookqa.compute_obqa_accuracy),
    "qnli": (qnli.get_train_dataloader, qnli.compute_qnli_accuracy),
    "multi_nli": (
        multi_nli.get_train_dataloader,
        multi_nli.compute_multi_nli_accuracy,
    ),
}


def get_dataset(name: str):
    try:
        return DATASETS[name.lower()]
    except KeyError as error:
        supported = ", ".join(DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Choose from: {supported}") from error
