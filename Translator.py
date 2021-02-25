import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import en_core_web_sm
import de_core_news_sm
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from transformer import Transformer
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


spacy_eng = en_core_web_sm.load()
spacy_ger = de_core_news_sm.load()


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

for i, example in enumerate([(x.src, x.trg) for x in train_data[0:5]]):
    print(f"Example_{i}:{example}")

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True
save_model = True

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
heads = 8
num_layers = 3
dropout = 0.10
max_length = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]
trg_pad_idx = english.vocab.stoi["<pad>"]
# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iter = BucketIterator(train_data, batch_size=batch_size, sort_within_batch=True, repeat=False,
                            sort_key=lambda x: len(x.src), device=device)

val_iter = BucketIterator(valid_data, batch_size=1, repeat=False, sort_within_batch=True, sort_key=lambda x: len(x.src),
                          device=device)

test_iter = BucketIterator(test_data, batch_size=1, repeat=False, sort_within_batch=True, sort_key=lambda x: len(x.src),
                           device=device)

batch = next(iter(train_iter))
trg_matrix = batch.trg.T
print(trg_matrix, trg_matrix.size())

for word in trg_matrix[3]:
    print(english.vocab.itos[word])

model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    embed_size,
    num_layers,
    forward_expansion,
    heads,
    dropout,
    device,
    max_length,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iter):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    # scheduler.step(mean_loss)

# running on entire test data takes a while
# score = bleu(test_data[1:100], model, german, english, device)
# print(f"Bleu score {score * 100:.2f}")
