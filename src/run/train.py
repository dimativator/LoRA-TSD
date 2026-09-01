import torch
from tqdm import tqdm

from .checkpointing import load_training_checkpoint, save_training_checkpoint


def _optimizer_parameters(optimizer):
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
        if parameter is not None
    ]


def train(config, model, optimizer, dataloader, tokenizer, evaluate):
    model.train()
    optimizer.zero_grad()
    resume_state = load_training_checkpoint(config, model, optimizer)
    losses = list(resume_state.get("losses", [])) if resume_state else []
    validation_scores = (
        list(resume_state.get("validation_scores", [])) if resume_state else []
    )
    global_step = int(resume_state["global_step"]) if resume_state else 0
    resume_epoch = int(resume_state["epoch"]) if resume_state else 0
    resume_batch_index = (
        int(resume_state["next_batch_index"]) if resume_state else 0
    )
    resume_generator_state = (
        resume_state.get("epoch_generator_state") if resume_state else None
    )
    progress = tqdm(
        total=config["steps"],
        initial=global_step,
        desc=config["run_name"],
    )

    epoch = resume_epoch
    first_epoch = True
    last_epoch = resume_epoch
    last_next_batch_index = resume_batch_index
    last_epoch_generator_state = resume_generator_state
    while global_step < config["steps"]:
        generator = getattr(dataloader, "generator", None)
        if first_epoch and resume_generator_state is not None and generator is not None:
            generator.set_state(resume_generator_state)
        epoch_generator_state = (
            generator.get_state() if generator is not None else None
        )

        data_iterator = iter(dataloader)
        start_batch_index = resume_batch_index if first_epoch else 0
        for _ in range(start_batch_index):
            try:
                next(data_iterator)
            except StopIteration:
                break

        micro_batches_since_update = 0
        for batch_index, batch in enumerate(
            data_iterator,
            start=start_batch_index,
        ):
            if global_step >= config["steps"]:
                break

            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss / config["grad_accum_steps"]
            loss.backward()
            micro_batches_since_update += 1

            if micro_batches_since_update < config["grad_accum_steps"]:
                continue

            if config["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    _optimizer_parameters(optimizer),
                    config["grad_clip_norm"],
                )

            optimizer.step()
            optimizer.zero_grad()
            micro_batches_since_update = 0
            global_step += 1
            last_epoch = epoch
            last_next_batch_index = batch_index + 1
            last_epoch_generator_state = epoch_generator_state
            loss_value = loss.item() * config["grad_accum_steps"]
            losses.append(loss_value)
            progress.update(1)
            progress.set_postfix(loss=f"{loss_value:.4f}")

            if (global_step - 1) % config["eval_every"] == 0:
                eval_batch_size = config.get("eval_batch_size")
                if eval_batch_size is None:
                    eval_batch_size = max(config["train_batch_size"] // 4, 1)
                score = evaluate(
                    model,
                    tokenizer,
                    eval_batch_size,
                )
                validation_scores.append({"step": global_step, "accuracy": score})
                print(f"validation_accuracy={score:.6f} step={global_step}", flush=True)
                model.train()

            checkpoint_every = int(config.get("checkpoint_every", 0) or 0)
            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                save_training_checkpoint(
                    config,
                    model,
                    optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    epoch_generator_state=epoch_generator_state,
                    losses=losses,
                    validation_scores=validation_scores,
                )

        epoch += 1
        first_epoch = False
        resume_batch_index = 0

    progress.close()
    if config.get("checkpoint_dir") or config.get("resume_from"):
        save_training_checkpoint(
            config,
            model,
            optimizer,
            global_step=global_step,
            epoch=last_epoch,
            next_batch_index=last_next_batch_index,
            epoch_generator_state=last_epoch_generator_state,
            losses=losses,
            validation_scores=validation_scores,
        )
    return losses, validation_scores
