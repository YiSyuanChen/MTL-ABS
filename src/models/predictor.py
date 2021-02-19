"""
Decoding implementation
"""
from __future__ import print_function
import codecs
import os
import torch
import mlflow
import pdb
from tensorboardX import SummaryWriter
from others.utils import rouge_results_to_str, test_rouge, tile

def build_predictor(args, tokenizer, symbols, model, logger=None):
    """ Build Predictor
    """
    translator = Translator(args,
                            model,
                            tokenizer,
                            symbols,
                            logger=logger)
    return translator


class Translator(object):
    """Uses a model to translate a batch of sentences.

    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """
    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 logger=None,
                 dump_beam=""):
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.vocab = vocab
        self.symbols = symbols
        self.logger = logger
        self.dump_beam = dump_beam

        self.generator = self.model.generator
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.tensorboard_writer = SummaryWriter(args.log_path, comment="Unmt")

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []
            }

    def translate(self, data_iter, step):
        """ Main control flow for decoding
        """

        # Set model to eval mode for decoding
        self.model.eval()

        # Output file path
        gold_path = os.path.join(self.args.result_path, 'test.%d.gold' % step)
        can_path = os.path.join(self.args.result_path,
                                'test.%d.candidate' % step)
        raw_src_path = os.path.join(self.args.result_path,
                                    'test.%d.raw_src' % step)
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                # batch (:obj:data_loader.Batch)
                # data_iter (:ojb:data_loader.Dataloader)

                # Constraint prediction length close to gold length
                if self.args.recall_eval:
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60

                # batch_data: type=dict
                #   keys -> ['predictions', 'scores', 'gold_score', 'batch']
                # translations: type=list
                #   content -> (predict_sent, gold_sent, raw_src)
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans

                    # type=string
                    src_str = src.strip()

                    # type=string
                    # [unused0] -> BOS
                    # [unused1] -> EOS
                    # [unused2] -> EOQ
                    pred_str = pred.replace('[unused0]', '').replace(
                        '[unused3]', '').replace('[PAD]', '').replace(
                            '[unused1]', '').replace(r' +', ' ').replace(
                                ' [unused2] ', '<q>').replace('[unused2]',
                                                              '').strip()
                    # type=string
                    gold_str = gold.strip()

                    # Constraint prediction length close to gold length
                    if (self.args.recall_eval):
                        _pred_str = ''
                        for sent in pred_str.split('<q>'):
                            # Accumulate pred_str sentence by sentnce
                            can_pred_str = _pred_str + '<q>' + sent.strip()

                            # Cut if length difference above 10 tokens
                            if (len(can_pred_str.split()) >=
                                    len(gold_str.split()) + 10):
                                pred_str = _pred_str
                                break
                            else:
                                _pred_str = can_pred_str

                    self.src_out_file.write(src_str + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    ct += 1

                # Flush the buffer
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        # Close files
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        # Report results in console and log
        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' %
                             (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F',
                                                   rouges['rouge_1_f_score'],
                                                   step)
                self.tensorboard_writer.add_scalar('test/rouge2-F',
                                                   rouges['rouge_2_f_score'],
                                                   step)
                self.tensorboard_writer.add_scalar('test/rougeL-F',
                                                   rouges['rouge_l_f_score'],
                                                   step)
                mlflow.log_metric('Test_ROUGE1_F', rouges['rouge_1_f_score'],
                                  step)
                mlflow.log_metric('Test_ROUGE2_F', rouges['rouge_2_f_score'],
                                  step)
                mlflow.log_metric('Test_ROUGEL_F', rouges['rouge_l_f_score'],
                                  step)

    def translate_batch(self, batch):
        """
        Translate a batch of sentences.
        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           fast (bool): enables fast beam search (may not support all features)
        """
        with torch.no_grad():
            return self._fast_translate_batch(batch,
                                              max_length=self.max_length,
                                              min_length=self.min_length)

    def _fast_translate_batch(self, batch, max_length, min_length=0):
        """ Main operation flow for decoding
        """
        # TODO: faster code path for beam_size == 1.
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        # Shared
        beam_size = self.beam_size
        batch_size = batch.batch_size
        device = batch.src.device

        # Generate encoder output
        # shape=(batch_size, src_len)
        src = batch.src
        # shape=(batch_size, src_len)
        segs = batch.segs
        # shape=(batch_size, src_len)
        mask_src = batch.mask_src
        # shape=(batch_size, src_len, emb_dim)
        src_features = self.model.bert(src, segs, mask_src)

        # Create dec_states
        dec_states = self.model.decoder.init_decoder_state(src,
                                                           src_features,
                                                           with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        # shape=(batch_size*beam_size, src_len, emb_dim)
        src_features = tile(src_features, beam_size, dim=0)

        # shape = (batch_size), content = tensor([0,1,2,...])
        batch_offset = torch.arange(batch_size,
                                    dtype=torch.long,
                                    device=device)

        # shape = (batch_size), content = tensor([0,beam_size,2*beam_size,...])
        beam_offset = torch.arange(0,
                                   batch_size * beam_size,
                                   step=beam_size,
                                   dtype=torch.long,
                                   device=device)

        # shape = (batch_size*beam_size), content = tensor([[1],[1],...])
        alive_seq = torch.full([batch_size * beam_size, 1],
                               self.start_token,
                               dtype=torch.long,
                               device=device)

        # Give full probability to the first beam on the first step.
        # shape = (batch_size*beam_size), content = tensor([0,-inf,..,0.-inf,...])
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] *
                                       (beam_size - 1),
                                       device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):

            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)

            # shape=(batch_size*beam_size, step)
            decoder_input = decoder_input.transpose(0, 1)

            # shape=(batch_size*beam_size, step, emb_dim)
            dec_out, dec_states = self.model.decoder(decoder_input,
                                                     src_features,
                                                     dec_states,
                                                     step=step)

            # Generator forward.
            # shape = (batch_size*beam_size, vocab_size)
            log_probs = self.generator.forward(
                dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            # Set the prob of end_token to min value to prevent stop
            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability. (Addition in log form)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            length_penalty = ((5.0 + (step + 1)) / 6.0)**self.args.alpha

            # Flatten probs into a list of possibilities.
            # shape = (batch_size*beam_size, vocab_size)
            curr_scores = log_probs / length_penalty

            # Avoid repeat trigram generations
            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(
                            alive_seq.size(0)):  # iterate batch_size*beam_size
                        fail = False

                        # id to word
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##', '').split()
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1])
                                    for i in range(1,
                                                   len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            # shape = (batch_size, beam_size)
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            # shape = (batch_size, beam_size)
            topk_beam_index = topk_ids.div(vocab_size)  # which beam
            # shape = (batch_size, beam_size)
            topk_ids = topk_ids.fmod(vocab_size)  # which token

            # Map beam_index to batch_index in the flat representation.
            # shape = (batch_size, beam_size)
            batch_index = (topk_beam_index +
                           beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            # shape = (batch_size*beam_size)
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat([
                alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)
            ], -1)

            # Check if end_token has been generated
            # shape=(batch_size,beam_size)
            is_finished = topk_ids.eq(self.end_token)

            # Stop decoding accord to max_length
            if step + 1 == max_length:
                is_finished.fill_(1)

            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                # shape=(batch_size,beam_size,tgt_len)
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):  # iterate batch_size

                    # if top beam is finished then set all beam for the data to finished
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)

                    # Store finished hypotheses (total score and predictions) for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:]))

                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b],
                                          key=lambda x: x[0],
                                          reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break

                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) == len(
            translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src = translation_batch[
            "predictions"], translation_batch["scores"], translation_batch[
                "gold_score"], batch.tgt_str, batch.src

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens(
                [int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##', '')
            gold_sent = ' '.join(tgt_str[b].split())
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            translations.append(translation)

        return translations

    def _report_rouge(self, gold_path, can_path):
        """  Calculate ROUGE scores
        """
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

