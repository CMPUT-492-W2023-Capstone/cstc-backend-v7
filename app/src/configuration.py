from enum import Enum
from pathlib import Path

import torch

from module import attempt_load, check_file, check_img_size,  LoadImages, LoadStreams
from utils.torch_utils import load_classifier, select_device, TracedModel


APP_PATH = Path(__file__).resolve().parents[0].parents[0]
MODELS_PATH = APP_PATH / 'models'
TRACKING_CONFIG = APP_PATH / 'tracking_configs'
VALID_URLs = ('rtsp://', 'rtmp://', 'https://')
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes


def _is_valid_file(source) -> bool:
    return Path(source).suffix[1:] in VID_FORMATS


def _is_valid_url(source) -> bool:
    return source.lower().startswith(VALID_URLs)


def _is_valid_webcam(source) -> bool:
    return source.isnumeric() or source.endswith('.txt')


class ComputingDevice(Enum):
    CUDA_0 = 0
    CUDA_1 = 1
    CUDA_2 = 2
    CUDA_3 = 3
    CPU = 'CPU'


class TrackingMethod(Enum):
    BOTSORT = 'botsort'
    BYTETRACK = 'bytetrack'
    DEEPOCSORT = 'deepocsort'
    OCSORT = 'ocsort'
    STRONGSORT = 'strongsort'


class InputConfig:

    def __init__(
            self,
            device=ComputingDevice.CPU.value,
            reid_models=MODELS_PATH / 'osnet_x0_25_msmt17.pt',
            media_source: str = '0',
            yolo_config: str = 'cfg/yolor_p6.cfg',
            yolo_models=MODELS_PATH / 'yolor_p6.pt'
    ):
        if type(device) is int:
            device = int(device)

        self.device = select_device(device)

        self.media_source = check_file(media_source) \
            if _is_valid_file(media_source) and _is_valid_url(media_source) else media_source

        self.reid_models = reid_models
        self.yolo_config = yolo_config
        self.yolo_models = yolo_models

        self.webcam_enable = _is_valid_webcam(media_source) or \
            (_is_valid_url(media_source) and not _is_valid_file(media_source))
        self.segmentation = self.yolo_models.name.endswith('-seg')

    def load_dataset_model(self, inference_img_size, fp16=False, trace=False):
        model = attempt_load(self.yolo_models, map_location=self.device)
        stride = int(model.stride.max())
        inference_img_size[0] = check_img_size(inference_img_size[0], s=stride)
        inference_img_size[1] = check_img_size(inference_img_size[1], s=stride)

        if trace:
            model = TracedModel(model, self.device, inference_img_size)
        if fp16:
            model.half()

        if self.webcam_enable:
            media_dataset = LoadStreams(self.media_source, img_size=inference_img_size, stride=stride)
        else:
            media_dataset = LoadImages(self.media_source, img_size=inference_img_size, stride=stride)

        return inference_img_size, media_dataset, len(media_dataset), model

    def load_classifier(self):
        model = load_classifier(name='resnet101', n=2)
        model.load_state_dict(
            torch.load(
                str(MODELS_PATH / 'resnet101.pt'),
                map_location=self.device)['model']
        )
        model.to(self.device).eval()

        return model


class OutputVideoConfig:

    def __init__(
            self,
            line_thickness: int = 2,
            hide_conf: bool = False,
            hide_class: bool = False,
            hide_labels: bool = False,
            show_video: bool = False,
            enable_trace: bool = False,
            vid_frame_stride: int = 1,  # number of frame will skip per second
            retina_masks: bool = False,
    ):
        self.line_thickness = line_thickness
        self.hide_conf = hide_conf
        self.hide_class = hide_class
        self.hide_labels = hide_labels
        self.show_video = show_video
        self.enable_trace = enable_trace
        self.vid_frame_stride = vid_frame_stride
        self.retina_masks = retina_masks


class OutputResultConfig:

    def __init__(
            self,
            no_save: bool = False,
            save_csv: bool = False,
            upload_period: int = 300,
    ):
        self.no_save = no_save
        self.save_csv = save_csv
        self.upload_period = upload_period


class AlgorithmConfig:

    def __init__(
            self,
            agnostic_nms: bool = False,
            augment: bool = False,
            blur_frames_limit: int = 50,
            blur_thersold: float = 100.0,
            classify: bool = False,
            class_filter: list = None,
            conf_thres: float = 0.25,
            device: ComputingDevice = ComputingDevice.CPU,
            dnn: bool = False,
            fp16: bool = False,
            inference_img_size: list = None,
            iou_thres: float = 0.5,
            max_det: int = 1000,
            tracking_method: TrackingMethod = TrackingMethod.BYTETRACK,
            tracking_config=TRACKING_CONFIG / 'bytetrack.yaml',
            velocity_thersold_delta: float = 0,
            update_period: int = 50
    ):
        if class_filter is None:
            class_filter = []
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.blur_frames_limit = blur_frames_limit
        self.blur_thersold = blur_thersold
        self.classify = classify
        self.class_filter = class_filter
        self.conf_thres = conf_thres
        self.device = device
        self.dnn = dnn
        self.fp16 = fp16
        self.iou_thres = iou_thres
        self.max_det = max_det

        if inference_img_size is None or len(inference_img_size) > 2 or len(inference_img_size) < 1:
            self.inference_img_size = [640, 640]
        else:
            self.inference_img_size = inference_img_size if len(inference_img_size) == 2 else 2 * inference_img_size

        self.tracking_method = tracking_method.value
        self.tracking_config = tracking_config
        self.velocity_thersold_delta = velocity_thersold_delta
        self.update_period = update_period
