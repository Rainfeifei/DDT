# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
# from detectron2.structures import pairwise_iou, Boxes


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def calculate_iou(box1, box2):
    # 计算两个边界框的相交区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算相交区域的面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou



def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
data_loader_gan, evaluator_gan: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        # for idx, inputs in enumerate(data_loader):
        for idx, (batch1, batch2) in enumerate(zip(data_loader, data_loader_gan)):

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
             # 存放图片的预测信息
            outputs = model(batch1)
            outputs_gan = model(batch2)

            target_box = outputs[0]['instances'].pred_boxes.tensor.cpu().numpy().tolist()
            target_scores = outputs[0]['instances'].scores.cpu().numpy().tolist()
            target_scores_pred_classes = outputs[0]['instances'].pred_classes.cpu().numpy().tolist()

            gan_box = outputs_gan[0]['instances'].pred_boxes.tensor.cpu().numpy().tolist()
            gan_scores = outputs_gan[0]['instances'].scores.cpu().numpy().tolist()
            gan_scores_pred_classes = outputs_gan[0]['instances'].pred_classes.cpu().numpy().tolist()
            num_target = len(outputs[0]['instances'].pred_boxes.tensor.cpu().numpy().tolist())
            num_gan = len(outputs_gan[0]['instances'].pred_boxes.tensor.cpu().numpy().tolist())

            new_box, new_scores, new_pred_classes = [], [], []

            # # 以target为基底，将source_like中未检测到底，而target中检测到的物体添加到source_like中去
            # for i in range(num_target):
            #     target_box_i = target_box[i]
            #     target_scores_i = target_scores[i]
            #     target_scores_pred_classes_i = target_scores_pred_classes[i]
            #     num = 0
            #     for j in range(num_gan):
            #         gan_box_j = gan_box[j]  #
            #         gan_scores_j = gan_scores[j]
            #         gan_scores_pred_classes_j = gan_scores_pred_classes[j]
            #         gan_box_test = [[18,19,2,30],[29,50,28,20]]
            #         target_box_test = [[18, 19, 2, 30], [29, 50, 28, 20]]
            #         iou = calculate_iou(gan_box_test, target_box_test)
            #         if iou > 0.45:
            #             num = num + 1
            #     if num != 0:
            #         continue
            #     else:
            #         new_box.append(target_box_i)
            #         new_scores.append(target_scores_i)
            #         new_pred_classes.append(target_scores_pred_classes_i)
            # outputs_gan[0]['instances'].pred_boxes.tensor = torch.cat((outputs_gan[0]['instances'].pred_boxes.tensor, torch.tensor(new_box).cuda()))
            # outputs_gan[0]['instances'].scores = torch.cat((outputs_gan[0]['instances'].scores, torch.tensor(new_scores).cuda()))
            # outputs_gan[0]['instances'].pred_classes = torch.cat((outputs_gan[0]['instances'].pred_classes, torch.tensor(new_pred_classes).cuda()))



            # 以gan为基底，target中的每个框和gan中的某一个物体计算iou值，如果其中有iou值大于某一个数值的，则gan中已经检测到了该物体，就
            # 不将gan中的物体加进来；如果gan和每一个target中的物体进行iou的比较，没有值大于某一数值，则target中没有检测到该物体，就将该gan中的
            # 物体加进来
            for i in range(num_gan):
                gan_box_i = gan_box[i]  #
                gan_scores_i = gan_scores[i]
                gan_scores_pred_classes_i = gan_scores_pred_classes[i]
                num = 0
                for j in range(num_target):
                    target_box_j = target_box[j]
                    target_scores_j = target_scores[j]
                    target_scores_pred_classes_j = target_scores_pred_classes[j]
                    iou =calculate_iou(gan_box_i, target_box_j)
                    # num = 0
                    if iou > 0.68:
                        num = num + 1

                if num != 0:
                    continue
                else:
                    new_box.append(gan_box_i)
                    new_scores.append(gan_scores_i)
                    new_pred_classes.append(gan_scores_pred_classes_i)

            # 将两组框合并
            outputs[0]['instances'].pred_boxes.tensor = torch.cat((outputs[0]['instances'].pred_boxes.tensor, torch.tensor(new_box).cuda()))
            outputs[0]['instances'].scores = torch.cat((outputs[0]['instances'].scores, torch.tensor(new_scores).cuda()))
            outputs[0]['instances'].pred_classes = torch.cat((outputs[0]['instances'].pred_classes, torch.tensor(new_pred_classes).cuda()))




            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()

            # 用于计算模型预测结果和真实标签之间的指标
            evaluator.process(batch2, outputs_gan)

            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
