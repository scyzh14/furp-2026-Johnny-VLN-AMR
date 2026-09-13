'''
Phase-5 validation: minimal Seq2Seq model check on PyTorch 2.7.

Loads the ORIGINAL R2R pipeline (no algorithm changes) and exercises, in
order, one full learning step:

    R2RBatch -> Seq2SeqAgent.rollout
             -> EncoderLSTM.forward          (instruction encoding)
             -> AttnDecoderLSTM.forward      (20 unrolled steps)
             -> SoftDotAttention             (masked with BoolTensor)
             -> nn.CrossEntropyLoss          (teacher-forced target)
             -> loss.backward()              (explicit, asserted gradients)
             -> optimizer.step()             (explicit, asserted update)

Run from the repo root inside the modern container:
    python3 tests/test_agent_forward.py

Model architecture, teacher-forcing logic, and loss math are untouched;
the only PyTorch-2.7-mandated change so far is mask.byte() -> mask.bool()
in agent.py (verified: masked_fill_ with uint8 raises in torch 2.7).
'''

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tasks', 'R2R'))
os.chdir(REPO_ROOT)

import torch
from torch import optim

from utils import read_vocab, Tokenizer
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM
from agent import Seq2SeqAgent

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
FEATURE_STORE = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80
BATCH_SIZE = 4   # >1: model.py squeeze() collapses the batch dim when batch==1

WORD_EMBEDDING_SIZE = 256
ACTION_EMBEDDING_SIZE = 32
HIDDEN_SIZE = 512
DROPOUT_RATIO = 0.5
MAX_EPISODE_LEN = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4


def main():
    print('torch %s (cuda %s, available=%s)' %
          (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
    assert torch.cuda.is_available()

    # ---- Data pipeline ----
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    env = R2RBatch(FEATURE_STORE, batch_size=BATCH_SIZE, splits=['train'], tokenizer=tok)
    print('[1] R2RBatch ready (%d instructions, batch=%d)' % (len(env.data), BATCH_SIZE))

    # ---- Original model (unmodified architecture) ----
    encoder = EncoderLSTM(len(vocab), WORD_EMBEDDING_SIZE, HIDDEN_SIZE,
                          padding_idx=tok.word_to_index['<PAD>'],
                          dropout_ratio=DROPOUT_RATIO, bidirectional=False).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              ACTION_EMBEDDING_SIZE, HIDDEN_SIZE,
                              DROPOUT_RATIO).cuda()
    agent = Seq2SeqAgent(env, results_path='', encoder=encoder, decoder=decoder,
                         episode_len=MAX_EPISODE_LEN)
    agent.feedback = 'teacher'   # original teacher-forcing baseline
    encoder.train(); decoder.train()
    n_enc_params = sum(p.numel() for p in encoder.parameters())
    n_dec_params = sum(p.numel() for p in decoder.parameters())
    print('[2] EncoderLSTM (%d params) + AttnDecoderLSTM (%d params) on %s'
          % (n_enc_params, n_dec_params, next(encoder.parameters()).device))

    # ---- Forward: rollout accumulates teacher-forced loss ----
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    agent.rollout()
    loss = agent.loss
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() > 0
    print('[3] forward + teacher-forced loss: %.4f' % loss.item())

    # ---- Backward ----
    loss.backward()
    enc_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    dec_grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0 and len(dec_grads) > 0, 'no gradients flowed'
    assert any(g.abs().sum().item() > 0 for g in enc_grads), 'encoder grads all zero'
    assert any(g.abs().sum().item() > 0 for g in dec_grads), 'decoder grads all zero'
    enc_norm = torch.sqrt(sum((g ** 2).sum() for g in enc_grads)).item()
    dec_norm = torch.sqrt(sum((g ** 2).sum() for g in dec_grads)).item()
    print('[4] backward OK: encoder grad-norm %.4f, decoder grad-norm %.4f' % (enc_norm, dec_norm))

    # ---- Optimizer step: parameters must actually change ----
    enc_before = next(encoder.parameters()).detach().clone()
    dec_before = next(decoder.parameters()).detach().clone()
    encoder_optimizer.step()
    decoder_optimizer.step()
    assert not torch.equal(enc_before, next(encoder.parameters()).detach()), 'encoder did not update'
    assert not torch.equal(dec_before, next(decoder.parameters()).detach()), 'decoder did not update'
    print('[5] optimizer.step: both optimizers updated parameters')

    # ---- Second iteration: student-forcing (sample) end-to-end sanity ----
    encoder_optimizer.zero_grad(); decoder_optimizer.zero_grad()
    agent.feedback = 'sample'
    agent.rollout()
    agent.loss.backward()
    encoder_optimizer.step(); decoder_optimizer.step()
    print('[6] second iteration with sample feedback: loss %.4f' % agent.loss.item())

    print('')
    print('ALL SEQ2SEQ MODEL TESTS PASSED '
          '(R2RBatch -> Agent -> Encoder -> Decoder -> Attention -> Loss -> Backward -> Step)')


if __name__ == '__main__':
    main()
