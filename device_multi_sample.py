import argparse
import json
import time

import jax
import numpy as np
import pandas as pd
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open
import wandb

from mesh_transformer.util import clip_by_global_norm


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    prompts_path = params['prompts']
    prompts = pd.read_csv(f'prompts/{prompts_path}')
    n_repeats = params['n_repeats']
    top_p = params.get("top_p", 0.9)
    temp = params.get("temp", 0.75)

    project = params.get("wandb_project", "mesh-transformer-jax")
    wandb.init(project=project, name=params["name"], config=params)
    prompt_table = wandb.Table(
        columns=['model_checkpoint','title', 'selection' ,'prompt', 'completion','top_p', 'temp', 'compleition_time'])
    prompt_df = pd.DataFrame()

    model_ls = []
    titles_ls = []
    selection_ls = []
    prompt_ls = []
    completion_ls = []
    top_p_ls = []
    temp_ls = []
    compleition_time_ls = []

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

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")
    
    total_batch = per_replica_batch * jax.device_count() // cores_per_replica
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        # while True:
        for i in range(len(prompts)):
            if not prompts.iloc[i].isnull().values.any():
                context = prompts.iloc[i]['first']
                title = prompts.iloc[i]['title']
                for rep in range(n_repeats):
                    # context = input("Type input:")
                    tokens = tokenizer.encode(context)

                    start = time.time()

                    provided_ctx = len(tokens)
                    pad_amount = seq - provided_ctx

                    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
                    batched_tokens = np.array([padded_tokens] * total_batch)
                    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

                    output = network.generate(batched_tokens, length, 512, {"top_p": np.ones(total_batch) * top_p,
                                                                            "temp": np.ones(total_batch) * temp})

                    for idx, o in enumerate(output[1][0][:, :, 0]):
                        print(f"sample {idx}: {repr(tokenizer.decode(o))}")

                    print(f"Title {i}: completion done in {time.time() - start:06}s \n")

                    prompt_table.add_data(
                        f'{model_dir}_{ckpt_step}', title, 'first' , context, repr(tokenizer.decode(o)),
                        top_p, temp, time.time() - start
                    )
                    model_ls.append(f'{model_dir}_{ckpt_step}')
                    titles_ls.append(title)
                    selection_ls.append('first')
                    prompt_ls.append(context)
                    completion_ls.append(str(repr(tokenizer.decode(o))))
                    top_p_ls.append(top_p)
                    temp_ls.append(temp)
                    compleition_time_ls.append(time.time() - start)
        
        wandb.log({'Prompt Table': prompt_table})

        prompt_df = pd.DataFrame({
            'model': model_ls,
            'title': titles_ls,
            'selection': selection_ls,
            'prompt': prompt_ls,
            'completion': completion_ls,
            'top_p': top_p_ls,
            'temp': temp_ls,
            'compleition_time': compleition_time_ls
        })

        from datetime import datetime
        now = datetime.now()

        prompt_df.to_csv(f"gs://{bucket}/{model_dir}/step_{ckpt_step}/prompt_completions_{now}.csv")
        wandb.finish()