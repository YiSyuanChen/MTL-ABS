""" Report manager utility """
from __future__ import print_function
from datetime import datetime

import time
import math
import sys

from distributed import all_gather_list
from others.logging import logger
import mlflow
import pdb


def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(opt.tensorboard_log_dir +
                               datetime.now().strftime("/%b-%d_%H-%M-%S"),
                               comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every,
                           start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """
    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self,
                        step,
                        num_steps,
                        learning_rate,
                        learning_rate_dec,
                        report_stats,
                        task_id=None,
                        multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            report_stats = Statistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_training(step,
                                  num_steps,
                                  learning_rate,
                                  learning_rate_dec,
                                  report_stats,
                                  task_id=task_id)
            self.progress_step += 1
        return Statistics()

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_inner_training(self,
                              step,
                              num_steps,
                              learning_rate,
                              learning_rate_dec,
                              report_stats,
                              task_id=None,
                              multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            report_stats = Statistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_inner_training(step,
                                        num_steps,
                                        learning_rate,
                                        learning_rate_dec,
                                        report_stats,
                                        task_id=task_id)
            self.progress_step += 1
        return Statistics()

    def _report_inner_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(lr,
                          step,
                          train_stats=train_stats,
                          valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer
        self.inner_step_accum = 0
        self.current_loss = float('inf')  # Big initial loss

    def maybe_log_tensorboard(self,
                              stats,
                              prefix,
                              learning_rate,
                              learning_rate_dec,
                              step,
                              task_id=None):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(prefix,
                                  self.tensorboard_writer,
                                  learning_rate,
                                  learning_rate_dec,
                                  step,
                                  task_id=task_id)

    def maybe_log_tensorboard_inner(self,
                                    stats,
                                    prefix,
                                    learning_rate,
                                    learning_rate_dec,
                                    step,
                                    task_id=None):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard_inner(prefix,
                                        self.tensorboard_writer,
                                        learning_rate,
                                        learning_rate_dec,
                                        step,
                                        task_id=task_id)

    def _report_training(self,
                         step,
                         num_steps,
                         learning_rate,
                         learning_rate_dec,
                         report_stats,
                         task_id=None):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps, learning_rate, learning_rate_dec,
                            self.start_time)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats,
                                   "Outer",
                                   learning_rate,
                                   learning_rate_dec,
                                   step,
                                   task_id=task_id)
        report_stats = Statistics()

        return report_stats

    def _report_inner_training(self,
                               step,
                               num_steps,
                               learning_rate,
                               learning_rate_dec,
                               report_stats,
                               task_id=None):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output_inner(step, num_steps, learning_rate,
                                  learning_rate_dec, self.start_time)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard_inner(report_stats,
                                         "Inner",
                                         learning_rate,
                                         learning_rate_dec,
                                         self.inner_step_accum + step,
                                         task_id=task_id)
        self.inner_step_accum += step
        self.current_loss = report_stats.current_loss
        report_stats = Statistics()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())

            self.maybe_log_tensorboard(train_stats, "train", lr, lr, step)
        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())

            self.maybe_log_tensorboard(valid_stats, "valid", lr, lr, step)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):

        self.loss = loss
        self.current_loss = loss
        self.n_words = n_words
        self.n_docs = 0
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.inner_step_accum = 0  # Add to clearly log inner step

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        from torch.distributed import get_rank
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        if (stat.n_words != 0):
            self.current_loss = stat.loss / stat.n_words  # handle for loss_KL update
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_docs += stat.n_docs

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        #return 100 * (self.n_correct / self.n_words)
        if (self.n_words != 0):
            return 100 * (self.n_correct / self.n_words)
        else:
            return 0

    def xent(self):
        """ compute cross entropy """
        #return self.loss / self.n_words
        if (self.n_words != 0):
            return self.loss / self.n_words
        else:
            return 0

    def ppl(self):
        """ compute perplexity """
        #return math.exp(min(self.loss / self.n_words, 100))
        if (self.n_words != 0):
            return math.exp(min(self.loss / self.n_words, 100))
        else:
            return 0

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, learning_rate_dec, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.8f; lr_dec: %7.8f; %3.0f/%3.0f tok/s; %6.0f sec") %
            (step, num_steps, self.accuracy(), self.ppl(), self.xent(),
             learning_rate, learning_rate_dec, self.n_src_words /
             (t + 1e-5), self.n_words / (t + 1e-5), time.time() - start))
        sys.stdout.flush()

    def output_inner(self, step, num_steps, learning_rate, learning_rate_dec,
                     start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            ("Inner Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.8f; lr_dec: %7.8f; %3.0f/%3.0f tok/s; %6.0f sec") %
            (step, num_steps, self.accuracy(), self.ppl(), self.xent(),
             learning_rate, learning_rate_dec, self.n_src_words /
             (t + 1e-5), self.n_words / (t + 1e-5), time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self,
                        prefix,
                        writer,
                        learning_rate,
                        learning_rate_dec,
                        step,
                        task_id=None):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
        writer.add_scalar(prefix + "/lr_dec", learning_rate_dec, step)

    def log_tensorboard_inner(self,
                              prefix,
                              writer,
                              learning_rate,
                              learning_rate_dec,
                              step,
                              task_id=None):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/inner/xent", self.xent(), step)
        writer.add_scalar(prefix + "/inner/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/inner/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/inner/lr", learning_rate, step)
        writer.add_scalar(prefix + "/inner/lr_dec", learning_rate_dec, step)
