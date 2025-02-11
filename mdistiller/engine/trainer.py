import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
    init_optimizer,
    init_scheduler,
    get_lr,
    setup_logger,
    log_training
)
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from ..distillers.KD import normalize

class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = init_optimizer(self.distiller.module, cfg)
        self.scheduler = init_scheduler(self.optimizer, cfg) 
        self.best_acc = -1
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        self.tf_writer = setup_logger(self.log_path)
        self.tqdm_leave = cfg.LOG.TQDM_LEAVE

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter), leave=self.tqdm_leave)

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller, self.tqdm_leave)

        if self.scheduler:
            self.scheduler.step()
        lr = get_lr(self.optimizer)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss
            }
        )
        self.best_acc = log_training(self.tf_writer, self.log_path, lr, epoch, log_dict, self.best_acc, self.cfg.LOG.WANDB)

        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.mean().cpu().detach().item(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class AugTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image_weak, image_strong = image
        image_weak, image_strong = image_weak.float(), image_strong.float()
        image_weak, image_strong = image_weak.cuda(non_blocking=True), image_strong.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image_weak=image_weak, image_strong=image_strong, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image_weak.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


# KLKDTrainer.py
class KLKDTrainer(BaseTrainer):
    """
    KLKDTrainer (Batch-level Sampling with Caching):
    
    - 매 미니 배치마다 enlarged batch에서 teacher와 student의 KL divergence를 계산하여,
      상위 target_batch_size 개의 샘플을 선택하고, 이때 선택된 데이터셋 고유 인덱스를 저장합니다.
    - sampling epoch(예: 설정한 주기마다)에는 매 미니 배치마다 선택을 수행하고, 전체 epoch에서 선택된 인덱스들의
      합집합을 캐시(self.cached_indices)로 저장합니다.
    - sampling epoch이 아닌 경우에는 캐시된 인덱스에 해당하는 데이터만을 사용하여 학습합니다.
    
    **주의사항:**
      - cfg에 아래 항목들이 있어야 합니다.
          cfg.TRAIN.BATCH_SIZE: 최종 학습에 사용할 배치 크기 (effective batch size)
          cfg.KLKD.START_RATE, cfg.KLKD.END_RATE: epoch에 따른 샘플 선택 비율 (예: 0.5 → 0.4)
          cfg.SOLVER.EPOCHS: 총 epoch 수
          cfg.KLKD.SAMPLING_PERIOD: sampling epoch 주기 (정수 혹은 리스트)
          (추가로, DATA.NUM_WORKERS, DATA.PIN_MEMORY 등이 있을 수 있음)
      - sampling epoch에서는 DataLoader가 enlarged batch를 제공해야 하며,
        비 sampling epoch에서는 캐시된 인덱스에 의한 SubsetRandomSampler를 사용합니다.
    """
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        super(KLKDTrainer, self).__init__(experiment_name, distiller, train_loader, val_loader, cfg)
        # 최종 학습에 사용될 배치 크기
        self.target_batch_size = cfg.SOLVER.BATCH_SIZE  
        # 시작 및 종료 시점의 K% (예: 0.5 → 0.4)
        self.start_rate = cfg.KLKD.START_RATE  
        self.end_rate = cfg.KLKD.END_RATE
        self.total_epochs = cfg.SOLVER.EPOCHS
        self.exclusion_rate = cfg.KLKD.EXCLUSION_RATE
        # Sampling 주기 (정수 또는 리스트)
        self.sampling_period_config = cfg.KLKD.SAMPLING_PERIOD

        # sampling epoch에 모은 각 미니 배치의 선택된 데이터셋 고유 인덱스를 임시 저장 (매 sampling epoch마다 초기화)
        self.epoch_cached_indices = []
        # 이전 sampling epoch에 모은 global selected indices (다음 epoch부터 사용)
        self.cached_indices = None

        # 현재 epoch이 sampling epoch인지 여부 플래그 (train_iter에서 분기 처리에 사용)
        self.do_sample = False

        # 원본 데이터셋과 DataLoader 재구성을 위해 저장
        self.train_dataset = self.train_loader.dataset

        # KD distiller 내부의 temperature 및 logit normalization 옵션
        self.temperature = distiller.module.temperature  
        self.logit_stand = distiller.module.logit_stand

        # 샘플개수 (디버깅용)
        self.sample_counter = 0

    def get_current_rate(self, epoch):
        """
        현재 epoch에 따른 샘플 선택 비율(K%)를 선형 보간하여 계산합니다.
          current_rate = start_rate + (end_rate - start_rate) * ((epoch - 1) / (total_epochs - 1))
        """
        return self.start_rate + (self.end_rate - self.start_rate) * ((epoch - 1) / (self.total_epochs - 1))

    def get_current_sampling_period(self, epoch):
        """
        cfg.KLKD.SAMPLING_PERIOD가 정수이면 그대로 사용하고,
        리스트인 경우 전체 epoch를 여러 구간으로 나누어 해당 구간의 주기를 반환합니다.
        """
        if isinstance(self.sampling_period_config, int):
            return self.sampling_period_config
        else:
            periods = self.sampling_period_config
            num_segments = len(periods)
            segment_length = self.total_epochs // num_segments
            segment_index = (epoch - 1) // segment_length
            if segment_index >= num_segments:
                segment_index = num_segments - 1
            return periods[segment_index]

    def is_sampling_epoch(self, epoch):
        """
        현재 epoch이 sampling을 수행해야 하는 epoch인지 결정합니다.
        (예: epoch 1부터 시작하여, 매 sampling_period마다 sampling)
        """
        period = self.get_current_sampling_period(epoch)
        return (epoch - 1) % period == 0

    def update_train_loader(self, epoch):
        """
        현재 epoch에 따라 DataLoader를 재구성합니다.
        - Sampling epoch인 경우 enlarged batch size (target_batch_size / current_rate)로 DataLoader를 생성하고,
          내부적으로 매 미니 배치마다 KL divergence를 계산하여 샘플링할 예정입니다.
        - 그 외의 epoch에서는 이전 sampling epoch에서 모은 cached_indices를 이용해 SubsetRandomSampler로 DataLoader를 생성합니다.
        """
        current_rate = self.get_current_rate(epoch)
        if self.is_sampling_epoch(epoch):
            self.do_sample = True
            # sampling epoch 시작 시, 이전에 모은 인덱스 초기화
            self.epoch_cached_indices = []
            # enlarged batch size: target_batch_size / current_rate (올림 처리)
            enlarged_bs = math.ceil(self.target_batch_size / current_rate)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=enlarged_bs,
                shuffle=True,   # sampling epoch에서는 전체 데이터를 섞어서 처리
                num_workers=2,
                pin_memory=True,
            )
            print(f"Epoch {epoch}: Sampling epoch with enlarged batch size {enlarged_bs}.")
        else:
            self.do_sample = False
            if self.cached_indices is not None and len(self.cached_indices) > 0:
                new_sampler = SubsetRandomSampler(self.cached_indices)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.target_batch_size,
                    sampler=new_sampler,
                    num_workers=2,
                    pin_memory=True,
                )
                print(f"Epoch {epoch}: Using cached {len(self.cached_indices)} sample indices.")
            else:
                # 캐시된 인덱스가 없으면 기본 DataLoader를 그대로 사용
                print(f"Epoch {epoch}: No cached indices found. Using default DataLoader.")

    def train_epoch(self, epoch):
        # 에폭 시작 전에 DataLoader를 업데이트합니다.
        self.update_train_loader(epoch)
        self.sample_counter = 0  # 샘플 카운터 초기화
        # 부모 클래스(BaseTrainer)의 train_epoch 호출 (여기서 self.train_loader를 순회함)
        ret = super(KLKDTrainer, self).train_epoch(epoch)
        # 만약 sampling epoch였다면, 에폭 종료 후 모은 인덱스들을 캐시합니다.
        if self.do_sample:
            # 중복 제거 후 리스트화 (순서는 크게 중요하지 않음)
            unique_indices = list(set(self.epoch_cached_indices))
            self.cached_indices = unique_indices
            print(f"Epoch {epoch}: Sampling complete. Cached {len(self.cached_indices)} unique indices for future epochs.")
        return ret

    def train_iter(self, data, epoch, train_meters):
        """
        - Sampling epoch인 경우:
            enlarged batch를 입력받아 teacher와 student의 KL divergence를 계산하고,
            상위 target_batch_size 개의 샘플을 선택하여 forward/backward 수행.
            선택된 샘플의 데이터셋 고유 인덱스는 self.epoch_cached_indices에 저장.
        - 비 sampling epoch인 경우:
            DataLoader가 이미 cached_indices에 기반한 effective batch를 제공하므로,
            바로 forward/backward를 수행합니다.
        """
        self.optimizer.zero_grad()

        # data unpack: images, targets, indices (indices: 데이터셋 고유 인덱스)
        images, targets, indices = data
        images = images.float().cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)

        # 기존 코드 (train_iter 내, sampling epoch 분기)
        if self.do_sample:
            # --- Sampling epoch: enlarged batch -> KL divergence 계산 후 상위 샘플 선택 ---
            # teacher와 student의 예측 계산
            student_logits, teacher_logits = self.distiller.module.get_logits(images)
            temperature = self.temperature
            log_pred_student = F.log_softmax(student_logits / temperature, dim=1)
            pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
            kl_div = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(dim=1) * (temperature ** 2)
            
            effective_bs = self.target_batch_size
            N = images.size(0)  # enlarged batch의 크기

            if N < effective_bs:
                selection = torch.arange(N).to(images.device)
            else:
                # 상위 exclusion_rate(예: 5%)는 제외하고, 그 다음 effective_bs개 선택하도록 함
                exclusion_rate = self.exclusion_rate  # 상위 5%를 제외
                # 내림차순 정렬: 가장 높은 KL divergence부터 정렬됨
                sorted_kl, sorted_indices = torch.sort(kl_div, descending=True)
                # 제외할 샘플 수 계산 (반올림)
                skip_num = int(math.ceil(N * exclusion_rate))
                # 선택할 인덱스 범위: skip_num부터 skip_num + effective_bs까지
                if skip_num + effective_bs > N:
                    # 만약 남은 샘플 수가 부족하면 가능한 만큼 선택 (필요시 추가 보충 로직 고려)
                    selection = sorted_indices[skip_num:]
                else:
                    selection = sorted_indices[skip_num: skip_num + effective_bs]
            
            # 선택된 인덱스(데이터셋 고유 인덱스)를 캐싱 (CPU로 옮겨 리스트에 추가)
            selected_dataset_indices = indices[selection].detach().cpu().tolist()
            self.epoch_cached_indices.extend(selected_dataset_indices)

            # 선택된 샘플로 forward/backward 수행
            sel_images = images[selection]
            sel_targets = targets[selection]
            preds, losses_dict = self.distiller(image=sel_images, target=sel_targets, epoch=epoch)
        else:
            # --- 비 sampling epoch: DataLoader가 이미 cached_indices를 기반으로 제공한 effective batch ---
            preds, losses_dict = self.distiller(image= images, target = targets, epoch = epoch)

        # Loss 계산 및 optimizer step
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()

        # 학습 지표 업데이트
        batch_size = images.size(0) if not self.do_sample else sel_images.size(0)
        acc1, acc5 = accuracy(preds, targets if not self.do_sample else sel_targets, topk=(1, 5))
        loss_val = loss.cpu().detach().item()
        train_meters["losses"].update(loss_val, batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)

        msg = ("Epoch:{} | Batch:{} | KLKD Loss:{:.4f} | Top-1:{:.3f} | Top-5:{:.3f}"
               .format(epoch, self.sample_counter, train_meters["losses"].avg, 
                       train_meters["top1"].avg, train_meters["top5"].avg))
        self.sample_counter += batch_size
        return msg


