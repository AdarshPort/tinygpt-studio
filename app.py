import math
import os

import streamlit as st
import torch
import torch.nn.functional as F

from model import GPTMini
import wandb


WANDB_ENTITY = "adarsh_d_mech-none"
WANDB_PROJECT = "tinygpt-from-scratch"
WANDB_RUN_ID = "0d6qfyb3"

if "WANDB_API_KEY" in st.secrets:
    os.environ["WANDB_API_KEY"] = st.secrets["WANDB_API_KEY"]

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="TinyGPT Studio",
    page_icon="🤖",
    layout="wide",
)

st.title("TinyGPT Studio")
st.write(
    "An interactive demo of a character-level GPT model trained from scratch "
    "with PyTorch."
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Load model and checkpoint
# =========================================================
@st.cache_resource
def load_model():
    checkpoint_path = "output/tinygpt_best.pt"

    if not os.path.exists(checkpoint_path):
        st.error(
            "No trained checkpoint found. Please train the model first or "
            "place `tinygpt_best.pt` in the `output/` folder."
        )
        st.stop()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = GPTMini(
        vocab_size=checkpoint["vocab_size"],
        block_size=config["block_size"],
        n_embed=config["n_embed"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


model, checkpoint = load_model()

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint["config"]


# =========================================================
# Tokenizer helpers
# =========================================================
def encode(text):
    return [stoi[char] for char in text]


def decode(token_ids):
    return "".join([itos[token_id] for token_id in token_ids])


# =========================================================
# Load evaluation data
# =========================================================
@st.cache_data
def load_text_data():
    data_path = "data/tiny-shakespeare.txt"

    if not os.path.exists(data_path):
        st.error(
            "Could not find `data/tinytext.txt`. Please add your training "
            "text file to the `data/` folder."
        )
        st.stop()

    with open(data_path, "r", encoding="utf-8") as file:
        return file.read()


raw_text = load_text_data()

chars = sorted(list(set(raw_text)))
evaluation_stoi = {char: index for index, char in enumerate(chars)}
evaluation_itos = {index: char for char, index in evaluation_stoi.items()}


def evaluation_encode(text):
    return [evaluation_stoi[char] for char in text]


def evaluation_decode(token_ids):
    return "".join([
        evaluation_itos[token_id]
        for token_id in token_ids
    ])


evaluation_data = torch.tensor(
    evaluation_encode(raw_text),
    dtype=torch.long,
)

n_total = len(evaluation_data)
n_train = int(0.80 * n_total)
n_val = int(0.10 * n_total)

test_data = evaluation_data[n_train + n_val:]


def get_test_batch(
    batch_size=32,
    block_size=None,
):
    if block_size is None:
        block_size = config["block_size"]

    if len(test_data) <= block_size + 1:
        st.error(
            "The test split is too small for the selected context length."
        )
        st.stop()

    indices = torch.randint(
        len(test_data) - block_size - 1,
        (batch_size,),
    )

    x = torch.stack([
        test_data[index:index + block_size]
        for index in indices
    ])

    y = torch.stack([
        test_data[index + 1:index + block_size + 1]
        for index in indices
    ])

    return x.to(device), y.to(device)


# =========================================================
# Evaluation function
# =========================================================
@torch.no_grad()
def evaluate_model(num_batches=100):
    model.eval()

    losses = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(num_batches):
        x, y = get_test_batch()

        _, loss = model(x, y)
        losses.append(loss.item())

        progress_bar.progress((step + 1) / num_batches)
        status_text.text(
            f"Evaluating batch {step + 1} of {num_batches}"
        )

    progress_bar.empty()
    status_text.empty()

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)

    return mean_loss, perplexity


# =========================================================
# Load wandb history
# =========================================================
@st.cache_data(ttl=300)
def load_wandb_history():
    api = wandb.Api()

    run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}"

    run = api.run(run_path)

    history = run.history(
        keys=[
            "train/loss",
            "eval/train_loss",
            "eval/val_loss",
            "eval/val_perplexity",
            "train/learning_rate",
        ],
    )

    return history


# =========================================================
# Tabs
# =========================================================
generation_tab, evaluation_tab, about_tab = st.tabs(
    [
        "Generate text",
        "Model evaluation",
        "About the model",
    ]
)


# =========================================================
# Tab 1: Generate text
# =========================================================
with generation_tab:
    st.header("Generate text")

    st.write(
        "Enter a prompt and generate text using the trained TinyGPT model. "
        "Temperature controls randomness, while top-k limits sampling to the "
        "most likely next characters."
    )

    col_input, col_settings = st.columns([2, 1])

    with col_input:
        prompt = st.text_area(
            "Prompt",
            value="Once upon a time",
            height=140,
        )

        generate_button = st.button(
            "Generate text",
            use_container_width=True,
        )

    with col_settings:
        max_new_tokens = st.slider(
            "Characters to generate",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.05,
            help=(
                "Lower values make output more predictable. "
                "Higher values make output more creative and random."
            ),
        )

        top_k = st.slider(
            "Top-k sampling",
            min_value=1,
            max_value=50,
            value=20,
            step=1,
            help=(
                "Limits each generation step to the k most likely "
                "next characters."
            ),
        )

    if generate_button:
        if not prompt:
            st.warning("Please enter a prompt.")
            st.stop()

        try:
            prompt_ids = torch.tensor(
                [encode(prompt)],
                dtype=torch.long,
                device=device,
            )
        except KeyError as error:
            st.error(
                f"The prompt contains a character not present in the "
                f"training vocabulary: {error}"
            )
            st.stop()

        with st.spinner("Generating text..."):
            generated_ids = model.generate(
                start_token_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            output_ids = generated_ids[0].tolist()
            generated_text = decode(output_ids)

        st.subheader("Generated output")

        st.text_area(
            "Model output",
            value=generated_text,
            height=350,
        )

        st.download_button(
            label="Download generated text",
            data=generated_text,
            file_name="tinygpt_generation.txt",
            mime="text/plain",
        )


# =========================================================
# Tab 2: Model evaluation
# =========================================================
with evaluation_tab:
    st.header("Model evaluation")

    st.write(
        "Evaluate the trained model on held-out test data. "
        "Lower cross-entropy loss and lower perplexity indicate better "
        "next-character prediction."
    )

    col_loss, col_perplexity = st.columns(2)

    with col_loss:
        st.metric(
            label="Checkpoint iteration",
            value=f"{checkpoint['iteration'] + 1}",
        )

    with col_perplexity:
        best_loss = checkpoint.get("best_val_loss")

        if best_loss is not None and best_loss != float("inf"):
            st.metric(
                label="Best validation loss",
                value=f"{best_loss:.4f}",
            )
        else:
            st.metric(
                label="Best validation loss",
                value="Not available",
            )

    
    st.subheader("Run test evaluation")

    num_evaluation_batches = st.slider(
        "Number of evaluation batches",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
    )

    if st.button(
        "Run evaluation",
        use_container_width=True,
    ):
        with st.spinner("Evaluating model on held-out test data..."):
            test_loss, test_perplexity = evaluate_model(
                num_batches=num_evaluation_batches,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Test loss",
                value=f"{test_loss:.4f}",
                help=(
                    "Average cross-entropy loss on held-out test data. "
                    "Lower is better."
                ),
            )

        with col2:
            st.metric(
                label="Test perplexity",
                value=f"{test_perplexity:.2f}",
                help=(
                    "Exponential of test loss. Lower means the model is "
                    "less uncertain about the next character."
                ),
            )

        st.info(
            "For this character-level model, perplexity is measured per "
            "character. For example, a perplexity of 10 means the model is "
            "roughly as uncertain as choosing among 10 equally likely next "
            "characters."
        )

    st.divider()
    
    st.subheader("Next-character probabilities")

    st.write(
        "Enter a prompt to inspect the model’s predicted probability "
        "distribution for the next character."
    )

    probability_prompt = st.text_input(
        "Prompt for probability analysis",
        value="Once upon a time",
    )

    if st.button(
        "Analyze next-character probabilities",
        use_container_width=True,
    ):
        if not probability_prompt:
            st.warning("Please enter a prompt.")
            st.stop()

        try:
            prompt_ids = torch.tensor(
                [encode(probability_prompt)],
                dtype=torch.long,
                device=device,
            )
        except KeyError as error:
            st.error(
                f"The prompt contains a character not present in the "
                f"training vocabulary: {error}"
            )
            st.stop()

        with torch.no_grad():
            logits, _ = model(prompt_ids)

        next_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=10,
        )

        top_characters = [
            itos[index.item()]
            for index in top_indices
        ]

        probability_chart_data = {
            "Next character": top_characters,
            "Probability": top_probabilities.cpu().numpy(),
        }

        st.bar_chart(
            probability_chart_data,
            x="Next character",
            y="Probability",
        )

        st.write(
            "These probabilities show which characters the model considers "
            "most likely to come next after the entered prompt."
        )
    st.divider()
    
    st.subheader("W&B training metrics")

    if st.button(
        "Load W&B training charts",
        use_container_width=True,
    ):
        try:
            wandb_history = load_wandb_history()
        except Exception as error:
            st.error(
                "Could not load W&B history. Please check your W&B entity, "
                "project, run ID, authentication, and internet connection."
            )
            st.exception(error)
            st.stop()

        if wandb_history.empty:
            st.warning(
                "No W&B history was found for the selected run."
            )
            st.stop()

        st.success(
            "W&B training history loaded successfully."
        )

        st.subheader("Training and validation loss")

        loss_chart_data = wandb_history[
            [
                "_step",
                "train/loss",
                "eval/train_loss",
                "eval/val_loss",
            ]
        ].copy()

        loss_chart_data = loss_chart_data.rename(
            columns={
                "_step": "Iteration",
                "train/loss": "Training loss",
                "eval/train_loss": "Evaluation training loss",
                "eval/val_loss": "Validation loss",
            }
        )

        st.line_chart(
            loss_chart_data,
            x="Iteration",
        )

        st.subheader("Validation perplexity")

        perplexity_chart_data = wandb_history[
            [
                "_step",
                "eval/val_perplexity",
            ]
        ].copy()

        perplexity_chart_data = perplexity_chart_data.rename(
            columns={
                "_step": "Iteration",
                "eval/val_perplexity": "Validation perplexity",
            }
        )

        st.line_chart(
            perplexity_chart_data,
            x="Iteration",
        )

        st.subheader("Learning-rate schedule")

        learning_rate_chart_data = wandb_history[
            [
                "_step",
                "train/learning_rate",
            ]
        ].copy()

        learning_rate_chart_data = learning_rate_chart_data.rename(
            columns={
                "_step": "Iteration",
                "train/learning_rate": "Learning rate",
            }
        )

        st.line_chart(
            learning_rate_chart_data,
            x="Iteration",
        )

        with st.expander("View raw W&B metrics"):
            st.dataframe(wandb_history)

   
    st.divider()


# =========================================================
# Tab 3: About the model
# =========================================================
with about_tab:
    st.header("About the model")

    st.write(
        "TinyGPT is a small character-level GPT-style transformer trained "
        "from scratch using PyTorch. It learns to predict the next character "
        "in a sequence and can generate text from a user-provided prompt."
    )

    st.subheader("Architecture")

    st.markdown(
        """
        - Character-level tokenization
        - Token embeddings
        - Positional embeddings
        - Causal multi-head self-attention
        - Feed-forward neural networks
        - Residual connections
        - Layer normalization
        - Linear language-model head
        """
    )

    st.subheader("Training configuration")

    training_config = {
        "Optimizer": "AdamW",
        "Learning rate": config.get("learning_rate", "Not available"),
        "Minimum learning rate": config.get(
            "min_learning_rate",
            "Not available",
        ),
        "Warmup iterations": config.get("warmup_iters", "Not available"),
        "Maximum iterations": config.get("max_iters", "Not available"),
        "Batch size": config.get("batch_size", "Not available"),
        "Context length": config.get("block_size", "Not available"),
        "Embedding dimension": config.get("n_embed", "Not available"),
        "Attention heads": config.get("n_head", "Not available"),
        "Transformer blocks": config.get("n_layer", "Not available"),
        "Dropout": config.get("dropout", "Not available"),
        "Gradient clipping": config.get("grad_clip", "Not available"),
    }

    st.json(training_config)

    st.subheader("Evaluation approach")

    st.markdown(
        """
        The model is evaluated using:

        - Cross-entropy loss
        - Perplexity
        - Held-out validation data
        - Held-out test data
        - Qualitative text-generation samples

        Perplexity is calculated as:

        `perplexity = exp(cross_entropy_loss)`

        Lower loss and lower perplexity indicate that the model is better at
        predicting the next character in unseen text.
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - This is a very small model trained on a limited dataset.
        - It may generate repetitive, inconsistent, or nonsensical text.
        - It does not have broad factual knowledge.
        - It is not aligned, safety-filtered, or suitable for production use.
        - Generated text may reflect biases present in the training corpus.
        - It should not be used for factual claims or important decisions.
        """
    )

    st.subheader("Intended use")

    st.markdown(
        """
        This project is intended for education, experimentation, and
        demonstration of transformer language-model training. It shows the
        complete workflow from data preparation and model implementation to
        training, evaluation, and interactive deployment.
        """
    )