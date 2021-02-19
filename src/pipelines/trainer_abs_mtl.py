import os
import copy
import distributed
import math
import numpy as np

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from tensorboardX import SummaryWriter

from models.reporter import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
import pdb


def _tally_parameters(model):
    """Returns the number of parameters."""
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def _tally_trainable_parameters(model):
    """Returns the number of trainable parameters."""
    n_params = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad == True])
    return n_params


def build_MTLtrainer(args, device_id, model, optims, optims_inner, loss):
    """Builds the trainer class for MTL

    Args:
        device_id (int): the GPU id to be used
        model (models.model_builder.MTLAbsSummarizer')
        optims (list[models.optimizers.Optimizer])
        optims_inner (list[models.optimizer.Optimizer])
        loss (models.loss.NMTLossCompute)

    Returns:
        A object in type pipelines.trainer_abs_mtl.MTLTrainer.
    """

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    print('gpu_rank %d' % gpu_rank)

    # Prepare tensorborad writer
    writer = SummaryWriter(args.log_path, comment="Unmt")

    # Prepare report manager
    report_manager = ReportMgr(args.report_every,
                               start_time=-1,
                               tensorboard_writer=writer)
    report_inner_manager = ReportMgr(args.report_inner_every,
                                     start_time=-1,
                                     tensorboard_writer=writer)

    # Prepare trainer
    trainer = MTLTrainer(args, model, optims, optims_inner, loss,
                         grad_accum_count, n_gpu, gpu_rank, report_manager,
                         report_inner_manager)

    # Show # of (trainable) parameters
    if (model):
        n_params = _tally_parameters(model)
        trainable_n_params = _tally_trainable_parameters(model)
        logger.info('Number of parameters: %d' % n_params)
        logger.info('Number of trainalbe parameters: %d' % trainable_n_params)

    return trainer


class MTLTrainer(object):
    """Controls the training process.

    Attributes:
        model (models.model_builder.MTLAbsSummarizer)
        optims (list[models.optimizers.Optimizer])
        optims_inner (list[models.optimizers.Optimizer])
        loss (models.loss.NMTLossCompute)
        grad_accum_count (int)
        n_gpu (int)
        gpu_rank (int)
        report_manager (models.reporter.ReportMgr)
        report_inner_manager (models.reporter.ReportMgr)
        device (string)
        save_checkpoint_steps (int)
    """
    def __init__(self,
                 args,
                 model,
                 optims,
                 optims_inner,
                 loss,
                 grad_accum_count=1,
                 n_gpu=1,
                 gpu_rank=1,
                 report_manager=None,
                 report_inner_manager=None):

        # Basic attributes.
        self.args = args
        self.model = model
        self.optims = optims
        self.optims_inner = optims_inner
        self.loss = loss
        self.grad_accum_count = grad_accum_count  # which is args.accum_count
        self.n_gpu = n_gpu  # which is args.world_size
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.report_inner_manager = report_inner_manager
        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        self.save_checkpoint_steps = args.save_checkpoint_steps

        # Set model in training mode.
        if model:
            self.model.train()

        assert grad_accum_count > 0

    def train(self, train_iter_fct):
        """Main training process of MTL.

        Args:
            train_iter_fct (function):
                return a instance of data.data_loader.MetaDataloader.
        """

        logger.info('Start training... (' + str(self.args.maml_type) + ')')
        step = self.optims[0]._step + 1  # resume the step recorded in optims
        true_sup_batchs = []
        true_qry_batchs = []
        accum = 0
        task_accum = 0
        sup_normalization = 0
        qry_normalization = 0

        # Dataloader
        train_iter = train_iter_fct()  # class Dataloader

        # Reporter and Statistics
        report_outer_stats = Statistics()
        report_inner_stats = Statistics()
        self._start_report_manager(start_time=report_outer_stats.start_time)

        # Current only support MAML
        assert self.args.maml_type == 'maml'

        # Make sure the accumulation of gradient is correct
        assert self.args.accum_count == self.args.num_batch_in_task

        while step <= self.args.train_steps:  # NOTE: Outer loop
            for i, (sup_batch, qry_batch) in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    # Collect batches (= self.grad_accum_count) as real batch
                    true_sup_batchs.append(sup_batch)
                    true_qry_batchs.append(qry_batch)

                    # Count non-padding words in bathces
                    sup_num_tokens = sup_batch.tgt[:, 1:].ne(
                        self.loss.padding_idx).sum()
                    qry_num_tokens = qry_batch.tgt[:, 1:].ne(
                        self.loss.padding_idx).sum()
                    sup_normalization += sup_num_tokens.item()
                    qry_normalization += qry_num_tokens.item()

                    accum += 1
                    if accum == self.args.num_batch_in_task:
                        task_accum += 1

                        #=============== Inner Update ================
                        # Sum-up non-padding words from multi-GPU
                        if self.n_gpu > 1:
                            sup_normalization = sum(
                                distributed.all_gather_list(sup_normalization))

                        inner_step = 1
                        while inner_step <= self.args.inner_train_steps:  # NOTE: Inner loop

                            # Compute gradient and update
                            self._maml_inner_gradient_accumulation(
                                true_sup_batchs, sup_normalization,
                                report_inner_stats, inner_step, task_accum)

                            # Call self.report_manager to report training process (if reach args.report_every)
                            report_inner_stats = self._maybe_report_inner_training(
                                inner_step, self.args.inner_train_steps,
                                self.optims_inner[task_accum -
                                                  1][0].learning_rate,
                                self.optims_inner[task_accum -
                                                  1][1].learning_rate,
                                report_inner_stats)

                            inner_step += 1
                            if inner_step > self.args.inner_train_steps:
                                break

                        #=============== Outer Update ================

                        # Sum-up non-padding words from multi-GPU
                        if self.n_gpu > 1:
                            qry_normalization = sum(
                                distributed.all_gather_list(qry_normalization))

                        # Compute gradient and update
                        self._maml_outter_gradient_accumulation(
                            true_qry_batchs, qry_normalization,
                            report_outer_stats, step, inner_step, task_accum)

                        if (task_accum == self.args.num_task):
                            # Calculate gradient norm
                            total_norm = 0.0
                            for p in self.model.parameters():
                                if (p.grad is not None):
                                    param_norm = p.grad.data.norm(2)
                                    total_norm += param_norm.item()**2
                            total_norm = total_norm**(1. / 2)

                        #===============================================

                        # Reset
                        true_sup_batchs = []
                        true_qry_batchs = []
                        accum = 0
                        sup_normalization = 0
                        qry_normalization = 0

                if (task_accum == self.args.num_task):

                    # Call self.report_manager to report training process(if reach args.report_every)
                    report_outer_stats = self._maybe_report_training(
                        step, self.args.train_steps,
                        self.optims[0].learning_rate,
                        self.optims[1].learning_rate, report_outer_stats)

                    # Reset
                    task_accum = 0

                    # Save
                    if (step % self.save_checkpoint_steps == 0
                            and self.gpu_rank == 0):
                        self._save(step)

                    # Check steps to stop
                    step += 1
                    if step > self.args.train_steps:
                        break

            # End for an epoch, reload and reset
            train_iter = train_iter_fct()

        self.report_manager.tensorboard_writer.flush(
        )  # force to output the log


    def _maml_inner_gradient_accumulation(self,
                                          true_batchs,
                                          normalization,
                                          report_stats,
                                          inner_step,
                                          task_accum,
                                          inference_mode=False):
        """Inner loop training.

        NOTE: 1. At the end of function, the adapter will be set to fast weights mode.
              2. This function does not require self.model.zero_grad(), since it does not use .backward()

        Args:
            true_batchs (list[data.data_loader.Batch])
            normalization (int):
                the number of non-padding tokens in the batch.
            report_stats (models.reporter.Statistics)
            inner_step (int):
                current inner loop step.
            task_accum (int):
                current task.
        """
        grad = None
        for batch in true_batchs:

            src = batch.src
            tgt = batch.tgt
            segs = batch.segs
            clss = batch.clss
            mask_src = batch.mask_src
            mask_tgt = batch.mask_tgt
            mask_cls = batch.mask_cls

            outputs, scores = self.model(src, tgt, segs, clss, mask_src,
                                         mask_tgt, mask_cls)
            loss, batch_stats = self.loss.monolithic_compute_loss_return(
                batch, outputs)

            # Compute gradient for adapter modules
            if (grad == None or self.grad_accum_count == 1):
                if inner_step == 1:
                    grad = torch.autograd.grad(loss.div(normalization),
                                               self.model._adapter_vars())
                else:
                    grad = torch.autograd.grad(
                        loss.div(normalization),
                        self.model._adapter_fast_weights())
            else:
                if inner_step == 1:
                    next_grad = torch.autograd.grad(loss.div(normalization),
                                                    self.model._adapter_vars())
                else:
                    next_grad = torch.autograd.grad(
                        loss.div(normalization),
                        self.model._adapter_fast_weights())
                grad = tuple([sum(x) for x in zip(grad, next_grad)])

            batch_stats.n_docs = int(src.size(0))
            report_stats.update(batch_stats)

            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    distributed.all_reduce_and_rescale_tensors(grad, float(1))

                if inner_step == 1:
                    # Compute update values with Adam
                    _, update_values_bert = self.optims_inner[
                        task_accum - 1][0].step(
                            self.model._adapter_vars_bert(),
                            grad,
                            inner_step=inner_step)
                    _, update_values_dec = self.optims_inner[
                        task_accum - 1][1].step(self.model._adapter_vars_dec(),
                                                grad[len(update_values_bert):],
                                                inner_step=inner_step)
                    update_values = update_values_bert + update_values_dec
                    # Compute new weights that maintain a differential path to preivous weights
                    fast_weights = list(
                        map(lambda p: p[1] + p[0],
                            zip(update_values, self.model._adapter_vars())))
                else:
                    # Compute update values with Adam
                    _, update_values_bert = self.optims_inner[
                        task_accum - 1][0].step(
                            self.model._adapter_fast_weights_bert(),
                            grad,
                            inner_step=inner_step)
                    _, update_values_dec = self.optims_inner[
                        task_accum - 1][1].step(
                            self.model._adapter_fast_weights_dec(),
                            grad[len(update_values_bert):],
                            inner_step=inner_step)
                    update_values = update_values_bert + update_values_dec
                    # Compute new weights that maintain a differential path to preivous weights
                    fast_weights = list(
                        map(
                            lambda p: p[1] + p[0],
                            zip(update_values,
                                self.model._adapter_fast_weights())))

        # update only after accum batches
        if self.grad_accum_count > 1:
            # Multi GPU gradient gather
            if self.n_gpu > 1:
                distributed.all_reduce_and_rescale_tensors(grad, float(1))

            if inner_step == 1:
                # Compute update values with Adam
                _, update_values_bert = self.optims_inner[
                    task_accum - 1][0].step(self.model._adapter_vars_bert(),
                                            grad,
                                            inner_step=inner_step)
                _, update_values_dec = self.optims_inner[
                    task_accum - 1][1].step(self.model._adapter_vars_dec(),
                                            grad[len(update_values_bert):],
                                            inner_step=inner_step)
                update_values = update_values_bert + update_values_dec
                # Compute new weights that maintain a differential path to preivous weights
                fast_weights = list(
                    map(lambda p: p[1] + p[0],
                        zip(update_values, self.model._adapter_vars())))
            else:
                # Compute update values with Adam
                _, update_values_bert = self.optims_inner[
                    task_accum - 1][0].step(
                        self.model._adapter_fast_weights_bert(),
                        grad,
                        inner_step=i_nner_step)
                _, update_values_dec = self.optims_inner[
                    task_accum - 1][1].step(
                        self.model._adapter_fast_weights_dec(),
                        grad[len(update_values_bert):],
                        inner_step=inner_step)
                update_values = update_values_bert + update_values_dec
                # Compute new weights that maintain a differential path to preivous weights
                fast_weights = list(
                    map(lambda p: p[1] + p[0],
                        zip(update_values,
                            self.model._adapter_fast_weights())))

        # Do not accumulate gradient in inference mode
        if (inference_mode):
            fast_weights = [w.data for w in fast_weights]
            for w in fast_weights:
                w.requires_grad = True

        # NOTE: Use new weights to perform following computation, the derivative path still maintained
        self.model._cascade_fast_weights_grad(fast_weights)

    def _maml_outter_gradient_accumulation(self, true_batchs, normalization,
                                           report_stats, step, inner_step,
                                           task_accum):
        """Outer loop training.

        NOTE: At the end of function, the adapters will be set to vars mode.

        Args:
            true_batchs (list[data.data_loader.Batch])
            normalization (int):
                the number of non-padding tokens in the batch.
            report_stats (models.reporter.Statistics)
            step (int):
                current outer loop step.
            inner_step (int):
                current inner loop step.
            task_accum (int):
                current task.
        """
        if self.grad_accum_count > 1 and task_accum == 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1 and task_accum == 1:
                self.model.zero_grad()

            src = batch.src
            tgt = batch.tgt
            segs = batch.segs
            clss = batch.clss
            mask_src = batch.mask_src
            mask_tgt = batch.mask_tgt
            mask_cls = batch.mask_cls

            outputs, scores = self.model(src, tgt, segs, clss, mask_src,
                                         mask_tgt, mask_cls)
            batch_stats = self.loss.monolithic_compute_loss_backprop(
                batch, outputs, normalization)

            batch_stats.n_docs = int(src.size(0))
            report_stats.update(batch_stats)

            if self.grad_accum_count == 1 and task_accum == self.args.num_task:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [
                        p.grad.data for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))

                for o in self.optims:
                    o.step()

        # Update only after accum batches
        if self.grad_accum_count > 1 and task_accum == self.args.num_task:
            # Multi GPU gradient gather
            if self.n_gpu > 1:
                grads = [
                    p.grad.data for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))

            for o in self.optims:
                o.step()

        # NOTE: Clean fast weight
        self.model._clean_fast_weights_mode()

    def validate(self, valid_iter_fct, step=0):
        """Main validation process of MTL.

        Args:
            train_iter_fct (function):
                return a instance of data.data_loader.MetaDataloader.
        """
        logger.info('Start validating...')

        step = 0
        ckpt_step = self.optims[0]._step  # resume the step recorded in optims
        true_sup_batchs = []
        true_qry_batchs = []
        accum = 0
        task_accum = 0
        sup_normalization = 0
        qry_normalization = 0

        # Dataloader
        valid_iter = valid_iter_fct()  # class Dataloader

        # Reporter and Statistics
        report_outer_stats = Statistics()
        report_inner_stats = Statistics()
        self._start_report_manager(start_time=report_outer_stats.start_time)

        # Make sure the accumulation of gradient is correct
        assert self.args.accum_count == self.args.num_batch_in_task

        while step <= self.args.train_steps:

            for i, (sup_batch, qry_batch) in enumerate(valid_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    # Collect batches (= self.grad_accum_count) as real batch
                    true_sup_batchs.append(sup_batch)
                    true_qry_batchs.append(qry_batch)

                    # Count non-padding words in bathces
                    sup_num_tokens = sup_batch.tgt[:, 1:].ne(
                        self.loss.padding_idx).sum()
                    qry_num_tokens = qry_batch.tgt[:, 1:].ne(
                        self.loss.padding_idx).sum()
                    sup_normalization += sup_num_tokens.item()
                    qry_normalization += qry_num_tokens.item()

                    # Gradient normalize for tasks
                    qry_normalization = qry_normalization * self.args.num_task

                    accum += 1
                    if accum == self.args.num_batch_in_task:
                        task_accum += 1

                        # NOTE: Clear optimizer state
                        self.optims_inner[task_accum -
                                          1][0].optimizer.clear_states()
                        self.optims_inner[task_accum -
                                          1][1].optimizer.clear_states()

                        #=============== Inner Update ================
                        # Sum-up non-padding words from multi-GPU
                        if self.n_gpu > 1:
                            sup_normalization = sum(
                                distributed.all_gather_list(sup_normalization))

                        inner_step = 1
                        while inner_step <= self.args.inner_train_steps:
                            # Compute gradient and update
                            self._maml_inner_gradient_accumulation(
                                true_sup_batchs,
                                sup_normalization,
                                report_inner_stats,
                                inner_step,
                                task_accum,
                                inference_mode=True)

                            # Call self.report_manager to report training process (if reach args.report_every)
                            report_inner_stats = self._maybe_report_inner_training(
                                inner_step, self.args.inner_train_steps,
                                self.optims_inner[task_accum -
                                                  1][0].learning_rate,
                                self.optims_inner[task_accum -
                                                  1][1].learning_rate,
                                report_inner_stats)

                            inner_step += 1
                            if inner_step > self.args.inner_train_steps:
                                break
                        #===============================================

                        #=============== Outer No Update ================
                        self.model.eval()

                        # Calculate loss only, no update for the initialization
                        self._valid(true_qry_batchs, report_outer_stats,
                                    ckpt_step)

                        # Clean fast weight
                        self.model._clean_fast_weights_mode()

                        self.model.train()
                        #===============================================

                        # Reset
                        true_sup_batchs = []
                        true_qry_batchs = []
                        accum = 0
                        sup_normalization = 0
                        qry_normalization = 0

                if (task_accum == self.args.num_task):

                    # Reset
                    task_accum = 0

                    # Check steps to stop
                    step += 1
                    if step > self.args.train_steps:
                        break

            # End for an epoch, reload & reset
            valid_iter = valid_iter_fct()

        # Report average result afer all validation steps
        self._report_step(0, ckpt_step,
                          valid_stats=report_outer_stats)  # first arg is lr
        self.report_manager.tensorboard_writer.flush(
        )  # force to output the log

        return report_outer_stats

    def _valid(self, true_batchs, report_stats, step):
        """Validation sub-process.

        Args:
            true_batchs (list[data.data_loader.Batch])
            report_stats (models.reporter.Statistics)
            step (int):
                current outer loop step.
        """

        # Set model in validating mode.
        self.model.eval()

        with torch.no_grad():
            for batch in true_batchs:
                src = batch.src
                tgt = batch.tgt
                segs = batch.segs
                clss = batch.clss
                mask_src = batch.mask_src
                mask_tgt = batch.mask_tgt
                mask_cls = batch.mask_cls

                outputs, _ = self.model(src, tgt, segs, clss, mask_src,
                                        mask_tgt, mask_cls)
                batch_stats = self.loss.monolithic_compute_loss(batch, outputs)
                report_stats.update(batch_stats)
            return report_stats

    def _save(self, step):
        """Saves args, model and outer optimizers"""
        real_model = self.model
        model_state_dict = real_model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path,
                                       'model_step_%d.pt' % step)

        # NOTE: the model would not overwritten if exist
        logger.info("Saving checkpoint %s" % checkpoint_path)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """Starts report manager to report training stats. """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
                self.report_inner_manager.start()
            else:
                self.report_manager.start_time = start_time
                self.report_inner_manager.start_time = start_time

    # [For Inner Loop]
    def _maybe_report_inner_training(self,
                                     step,
                                     num_steps,
                                     learning_rate,
                                     learning_rate_dec,
                                     report_inner_stats,
                                     task_id=None):
        """Reports inner loop training stats."""
        if self.report_inner_manager is not None:
            return self.report_inner_manager.report_inner_training(
                step,
                num_steps,
                learning_rate,
                learning_rate_dec,
                report_inner_stats,
                task_id=task_id,
                multigpu=self.n_gpu > 1)

    # [For Outer Loop]
    def _maybe_report_training(self,
                               step,
                               num_steps,
                               learning_rate,
                               learning_rate_dec,
                               report_stats,
                               task_id=None):
        """Reports outer loop training stats."""
        if self.report_manager is not None:
            return self.report_manager.report_training(step,
                                                       num_steps,
                                                       learning_rate,
                                                       learning_rate_dec,
                                                       report_stats,
                                                       task_id=task_id,
                                                       multigpu=self.n_gpu > 1)


    # [For Validation]
    def _report_step(self,
                     learning_rate,
                     step,
                     train_stats=None,
                     valid_stats=None):
        """Simple function to report stats (if report_manager is set)"""
        if self.report_manager is not None:
            return self.report_manager.report_step(learning_rate,
                                                   step,
                                                   train_stats=train_stats,
                                                   valid_stats=valid_stats)

    # [For Multi-GPU Training]
    def _maybe_gather_stats(self, stat):
        """Gathers statistics in multi-processes cases. """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat
