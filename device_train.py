import argparse
import json
import time
from datetime import datetime

import jax
import numpy as np
import optax
import pandas as pd

import wandb
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from tfrecord_loader import TFRecordNewInputs
from smart_open import open
from google.cloud import storage
from google.cloud.exceptions import NotFound

from mesh_transformer.util import clip_by_global_norm, additive_weight_decay


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
    To use, download the full checkpoint archive, extract and upload to a GCS bucket, and set that as --tune-model-path
    Modify the config file:
        - set `model_dir` to where the checkpoints should be written during training
        - set `train_set`, `val_set` to index files for your data
        - set `tpu_size` to 8 (if on a v3-8)
        - set `warmup_steps`, `anneal_steps`, `lr`, `end_lr` to the lr schedule for your finetuning run
        - the global step will reset to 0, keep that in mind when writing your lr schedule
        - set `name` to specify the name of the Weights & Biases run
        - set `wandb_project` to specify the Weights & Biases project to log to
    To prepare data in the expected data format:
        - use the script `create_finetune_tfrecords.py` in this repo to create data in the expected format
        - upload the .tfrecords files to GCS
        - save their GCS paths to a index file under `data/`, see existing files for examples
    """,
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--tune-model-path", type=str, default=None, help="Base model to finetune")
    parser.add_argument("--fresh-opt", default=False, action="store_true", help="Use a newly initialized optimizer, ignoring any optimizer state saved in the base checkpoint")
    parser.add_argument("--log-samples", default=False, action="store_true", help="Generate text samples and log to Weights & Biases at every validation, text prompts specificed in config file")

    args = parser.parse_args()
    return args


def save(network, step, bucket, path, mp, aux=None, keep_n=3, delete_old=True):
    assert path
    client = storage.Client()

    if aux is None:
        aux = {}

    try:
        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)
    except:
        # create metadata file
        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    for shard_id in range(mp):
        write_ckpt(network.state, f"gs://{bucket}/{path}/step_{step}/", shard_id)

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                # print(f"deleting {blob.name}")
                assert path in blob.name
                blob.delete()
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        json.dump(meta, f)


def train_step(network, data):
    inputs = {
        "obs": data[:, :, :-1],
        "target": data[:, :, 1:],
    }

    loss, last_loss, grad_norm, grad_norm_micro = network.train(inputs)

    return (
        np.array(loss).mean(),
        np.array(last_loss).mean(),
        np.array(grad_norm).mean(),
        np.array(grad_norm_micro).mean(),
    )


def eval_step(network, data):
    inputs = {
        "obs": data[:, :-1],
        "target": data[:, 1:],
    }

    out = network.eval(inputs)
    loss = out["loss"]

    return np.array(loss).mean()


def log_samples(network, tokenizer, prompts_df, wandb_table, 
    n_repeats, top_p, temp, total_batch, seq, model_dir, step ,generations_path):
    # for each prompt given
    for i in range(len(prompts_df)):
        if not prompts_df.iloc[i].isnull().values.any():
            context = prompts_df.iloc[i, 0]
            
            # if n_repeats > 1 then multiple generations will be run for each prompt
            for _ in range(n_repeats):
                tokens = tokenizer.encode(context)
                start = time.time()
                provided_ctx = len(tokens)
                pad_amount = seq - provided_ctx

                padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
                batched_tokens = np.array([padded_tokens] * total_batch)
                length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

                output = network.generate(batched_tokens, length, 512, 
                    {
                    "top_p": np.ones(total_batch) * top_p,
                    "temp": np.ones(total_batch) * temp
                    }
                )

                for idx, o in enumerate(output[1][0][:, :, 0]):
                    completion = repr(tokenizer.decode(o))
                    print(f"sample {idx}: {completion}")

                completion_time = f'{time.time() - start:06}'
                print(f"Title {i}: completion done in {completion_time}s \n")

                wandb_table.add_data(
                    i, f'{model_dir}_{step}', context, completion, 
                    top_p, temp, completion_time
                )

                model_ls.append(f'{model_dir}_{step}')
                prompt_idx_ls.append(i)
                prompt_ls.append(context)
                completion_ls.append(str(completion))
                top_p_ls.append(top_p)
                temp_ls.append(temp)
                compleition_time_ls.append(completion_time)
    
    # save completiion to csv
    output_df = pd.DataFrame({
        'id': v,
        'model': model_ls,
        'prompt': prompt_ls,
        'completion': completion_ls,
        'top_p': top_p_ls,
        'temp': temp_ls,
        'compleition_time': compleition_time_ls
    })
    now = datetime.now()
    output_df.to_csv(f"{generations_path}/text_generations_{step}_t{temp}_tp{top_p}_{now}.csv")

    return wandb_table


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    eval_tasks = params["eval_harness_tasks"]
    total_steps = params["total_steps"]

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]

    # alpha parameter for the exponential moving averages used to compute B_simple
    noise_scale_alpha = params.get("noise_scale_alpha", 0.01)

    scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(scheduler)
    )

    params["optimizer"] = opt

    start = time.time()
    tpu_size = jax.device_count()
    if tpu_size < cores_per_replica:
        msg = f"each shard needs a separate device, but device count ({tpu_size}) < shard count ({cores_per_replica})"
        raise ValueError(msg)
    print(f"jax devices: {tpu_size}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    # pick initial ckpt - based on tuning vs train from scratch

    step = 0
    initial_ckpt_state_path = None
    train_loader = None

    if args.tune_model_path:
        print('`--tune_model_path` passed: we are beginning a fine-tuning run')
        fine_tuning = True
        initial_ckpt_state_path = args.tune_model_path
    else:
        print('`--tune_model_path` not passed: we are continuing a fine-tuning run from a checkpoint (or we are not fine-tuning)')
        fine_tuning = False
        initial_ckpt_model_dir = model_dir
        initial_ckpt_path = f"gs://{bucket}/{initial_ckpt_model_dir}"
        meta_path = f"{initial_ckpt_path}/meta.json"

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ckpt_step = meta["checkpoints"][-1]
            initial_ckpt_state_path = f"{initial_ckpt_path}/step_{ckpt_step}/"
            print(f"state will be restored from checkpoint {ckpt_step}")

            step = ckpt_step
            train_loader = meta['aux'][str(ckpt_step)].get("train_loader", None)
        except NotFound:
            # no checkpoint, start at zero
            print(f"No checkpoint to load at {initial_ckpt_path}. Training from scratch.")

    if initial_ckpt_state_path:
        print(f"path to load checkpoint from: {initial_ckpt_state_path}")
    else:
        print("not loading from a checkpoint")

    # set up datasets
    print("setting up datasets")

    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                          gradient_accumulation_steps,
                                          per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'],
                                      restore_state=train_loader)

    global_val_batch = per_replica_batch * tpu_size // cores_per_replica

    val_sets = {}

    for k, v in params["val_set"].items():
        val_sets[k] = TFRecordNewInputs(
            f"data/{v}", batch_size=(global_val_batch,), sample_size=seq
        )

    # tok/sec metrics
    sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    tokens_per_step = params['seq'] * sequences_per_step

    # load + run
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        print("initializing network")
        network = CausalTransformer(params)

        if initial_ckpt_state_path:
            print("loading network")
            if fine_tuning:
                # get the scheduler step stored in the just-initialized optimizer
                # should be zero
                init_sched_state = network.state["opt_state"][-1]

            start = time.time()
            network.state = read_ckpt(network.state, initial_ckpt_state_path, devices.shape[1], load_opt=(not args.fresh_opt))

            if fine_tuning:
                # overwrite the loaded scheduler step with zeros
                # this makes fine-tuning use the lr schedule in
                network.state["opt_state"][-1] = init_sched_state

            print(f"network loaded in {time.time() - start:.06}s")

        print('compiling train fn')
        start = time.time()
        loss, last_loss, grad_norm, grad_norm_micro = train_step(
            network, train_dataset.get_samples()
        )
        step += 1
        print(f"Train fn compiled in {time.time() - start:.06}s")

        print('compiling eval fn')
        start = time.time()
        for val_set in val_sets.values():
            eval_step(network, val_set.get_samples())
            val_set.reset()
        print(f"Eval fn compiled in {time.time() - start:.06}s")

        project = params.get("wandb_project", "mesh-transformer-jax")
        wandb.init(project=project, name=params["name"], config=params)
        
        # Log text generations to wandb during validation. Prompt text(s)
        # are passed via a csv file
        if args.log_samples:
            from mesh_transformer.sampling import nucleaus_sample
            params["sampler"] = nucleaus_sample
            n_repeats = params.get('n_repeats',1)
            top_p = params.get("top_p", 0.9)
            temp = params.get("temp", 0.75)

            # read prompts and create path to save generations to
            from pathlib import Path
            prompts_path = params['prompts_path']
            prompts_df = pd.read_csv(f'gs://{bucket}/{prompts_path}') 
            generations_path = Path(f"gs://{bucket}/{model_dir}/generated_samples")
            generations_path.mkdir(parents=True, exist_ok=True) 

            # create W&B Table to log generations to
            text_samples_table = wandb.Table(
                columns=['id', 'model_checkpoint' , 'prompt', 'completion','top_p', 'temp', 'compleition_time'])
            
            # prepare lists to log generations
            model_ls = []
            prompt_idx_ls = []
            prompt_ls = []
            completion_ls = []
            top_p_ls = []
            temp_ls = []
            compleition_time_ls = []

            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        G_noise_avg = None
        S_noise_avg = None

        while True:
            if (step % ckpt_every == 1) or step == total_steps:
                print(f"saving a checkpoint for step {step}")
                save(network, step, bucket, model_dir,
                     mp=cores_per_replica,
                     aux={"train_loader": train_dataset.get_state()},
                     delete_old=True,
                     )

            if step % val_every == 1:  # 1 because we've already taken a step to compile train fn
                for name, val_set in val_sets.items():
                    val_loss = []
                    for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                     desc=f"validation for step {step}, set {name}",
                                     total=val_batches):
                        val_loss.append(eval_step(network, i))
                    val_set.reset()

                    val_loss = np.array(val_loss).mean()
                    print(f"validation loss for step {step}, set {name}: {val_loss}")

                    wandb_stats = {f'val/loss_{name}': float(val_loss)}

                # If logging text generations during valiation to wandb
                if args.log_samples:
                    table = log_samples(network, tokenizer, prompts_df, text_samples_table, 
                        n_repeats, top_p, temp, global_val_batch, seq, model_dir, step, generations_path)
                    wandb_stats.update({'train_samples/text_samples_table': table})
                
                wandb.log(wandb_stats)

            if step == total_steps:
                print("training completed!")
                exit()

            start = time.time()
            loss, last_loss, grad_norm, grad_norm_micro = train_step(
                network, train_dataset.get_samples()
            )
            step += 1

            steps_per_sec = 1 / (time.time() - start)
            tokens_per_sec = tokens_per_step * steps_per_sec

            sequences_processed = sequences_per_step * step
            tokens_processed = tokens_per_step * step

            ### compute summary stats about the gradient

            # converts from grads-summed-over-microbatch (what `CasualTransformer.train` computes)
            # to grads-averaged-over-microbatch (what we want)
            #
            # (when taking gradient steps, the same conversion happens inside the optimizer
            #  via optax.scale(1 / gradient_accumulation_steps))
            grad_norm = grad_norm / gradient_accumulation_steps

            # compute G_noise and S_noise
            # from "An Empirical Model of Large-Batch Training" Appendix A.1
            # here, B_big = gradient_accumulation_steps, and B_small = 1 for convenience
            gbsmall = grad_norm_micro ** 2
            gbbig = grad_norm ** 2
            G_noise = (gradient_accumulation_steps * gbbig - gbsmall) / (
                gradient_accumulation_steps - 1
            )
            S_noise = (gbsmall - gbbig) / (1 - 1 / gradient_accumulation_steps)

            noise_scale_stats = {
                "noise/G_noise": G_noise,
                "noise/S_noise": S_noise,
            }

            # heuristic to avoid reporting G_noise in very early training when gradients are large
            # (these take a long time to wash out of the moving average that defines B_simple)
            use_step_in_noise_avgs = gbbig < 2

            if use_step_in_noise_avgs:
                # compute moving averages of G_noise and S_noise, for B_simple
                if G_noise_avg is None:
                    G_noise_avg = G_noise
                else:
                    G_noise_avg = (1 - noise_scale_alpha) * G_noise_avg + noise_scale_alpha * G_noise

                if S_noise_avg is None:
                    S_noise_avg = S_noise
                else:
                    S_noise_avg = (1 - noise_scale_alpha) * S_noise_avg + noise_scale_alpha * S_noise

                B_simple = S_noise_avg / G_noise_avg

                noise_scale_stats.update(
                    {
                        "noise/G_noise_avg": G_noise_avg,
                        "noise/S_noise_avg": S_noise_avg,
                        "noise/B_simple": B_simple,
                    }
                )

            wandb_stats = {
                "train/loss": loss,
                "train/last_loss": last_loss,
                "train/steps_per_sec": steps_per_sec,
                "train/tokens_per_sec": tokens_per_sec,
                "train/grad_norm": grad_norm,
                "train/learning_rate": float(scheduler(network.state["opt_state"][-1].count[0].item())),
                "sequences_processed": sequences_processed,
                "tokens_processed": tokens_processed,
            }
            wandb_stats.update(noise_scale_stats)
            wandb.log(wandb_stats, step)
