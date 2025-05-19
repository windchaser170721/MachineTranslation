import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm

import config
from beam_decoder import beam_search
from utils import chinese_tokenizer_load


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, criterion, optimizer):
    best_bleu_score = -1
    early_stop = 0
    for epoch in range(1, config.epoch_num + 1):
        
        model.train()
        train_loss = run_epoch(train_data, model,
                               LossCompute(model.generator, criterion, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        
        model.eval()
        dev_loss = run_epoch(dev_data, model,
                             LossCompute(model.generator, criterion, None))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            logging.info("-------- Save Best Model! --------")
        
        early_stop += 1
        if early_stop > config.early_stop and bleu_score+config.bleu_gap < best_bleu_score:
            logging.info("-------- Early Stop! --------")
            break



class LossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


def evaluate(data, model, mode='dev'):
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        
        for batch in tqdm(data):
            
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        test_loss = run_epoch(data, model,
                              LossCompute(model.generator, criterion, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=False):
    sp_chn = chinese_tokenizer_load()
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
        
        decode_result = [h[0] for h in decode_result]
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        print(translation[0])
