import asyncio
import platform
from jsonargparse import CLI
from pathlib import Path

import cv2
import torch

import upload

from utils.general import apply_classifier, scale_coords

from ultralytics.yolo.utils.plotting import Annotator, colors

from algorithm import DetectionTask, TrackingTask
from configuration import InputConfig, OutputVideoConfig, OutputResultConfig, AlgorithmConfig


class TrackedObject:

    def __init__(self, v_id: int = -1, vehicle_type: str = '', confidence: int = -1, bbox: list = None):
        self.v_id = v_id
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.bbox = bbox

    def velocity(self):
        pass

    def get_label(self, config: OutputVideoConfig):
        label = 'Empty'
        if not config.hide_labels:
            class_label = None if config.hide_class else f'{self.vehicle_type}'
            confidence_label = None if config.hide_conf else f'{self.confidence:.2f}'

            label = f'{self.v_id} {class_label} {confidence_label}'

        return label

    """
    Check whether if two objects overlap each other
    bbox is x1', y1', x2', and y2'
    self is x1 , y1 , x2 , and y2
    """
    def is_overlap(self, bbox):
        return (self.bbox[0] <= bbox[2] and bbox[0] <= self.bbox[2]) and (  # X overlaps
                    self.bbox[1] <= bbox[3] and bbox[1] <= self.bbox[3])  # Y overlaps

    def update(self, vehicle_type: str = '', confidence: int = -1, bbox: list = None):
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.bbox = bbox

    def __str__(self):
        return f'Vehicle Id: {self.v_id} | Vehicle Type: {self.vehicle_type} ' \
               f'| Confidence: {self.confidence} | (X1, Y1, X2, Y2): ' \
               f'({self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]})'

    def __repr__(self):
        return self.__str__()


def stream_result(box_annotator: Annotator, im0, source_path, stream_windows: list):
    im0 = box_annotator.result()

    if platform.system() == 'Linux' and source_path not in stream_windows:
        stream_windows.append(source_path)

        cv2.namedWindow(str(source_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(str(source_path), im0.shape[1], im0.shape[0])

    cv2.imshow(str(source_path), im0)

    if cv2.waitKey(1) == ord('q'):
        exit()


def main(
        input_config: InputConfig,
        output_video_config: OutputVideoConfig,
        output_result_config: OutputResultConfig,
        algorithm_config: AlgorithmConfig
):
    # save_dir = save_options.configure_save(input_options.yolo_models)

    # load video frames (media dataset) and AI model
    algorithm_config.inference_img_size, media_dataset, media_dataset_size, model = \
        input_config.load_dataset_model(
            algorithm_config.inference_img_size,
            fp16=algorithm_config.fp16,
            trace=output_video_config.enable_trace
        )

    # names of the classification defined in the AI model
    class_names = model.module.names if hasattr(model, 'module') else model.names

    # save_video_paths, video_writes, save_txt_paths = [[None] * dataset_size for i in range(3)]

    seen_obj: int = 0
    streaming_windows: list = []

    current_frames, prev_frames = [[None] * media_dataset_size for i in range(2)]

    detection_task = DetectionTask(input_config, algorithm_config, output_result_config)
    tracking_task = TrackingTask(
        media_dataset_size,
        algorithm_config.tracking_method,
        algorithm_config.tracking_config,
        input_config.reid_models,
        input_config.device,
        algorithm_config.fp16
    )

    classifier = None
    if algorithm_config.classify:
        classifier = input_config.load_classifier()

    results = [[]] * media_dataset_size

    vehicles: {str: TrackedObject} = {}

    # For warmup
    old_img_w = algorithm_config.inference_img_size[0]
    old_img_h = algorithm_config.inference_img_size[1]
    old_img_b = 1

    if input_config.device != 'cpu':
        model(
            torch
            .zeros(1, 3, *algorithm_config.inference_img_size)
            .to(input_config.device)
            .type_as(next(model.parameters()))
        )

    log_msg = ''

    # OpenCV convention : im means image after modify, im0 means copy
    # of the image before modification (i.e. original image)
    for frame_index, batch in enumerate(media_dataset):
        source_paths, im, im0s, video_capture = batch
        
        detection_objs, im, old_img_w, old_img_h, old_img_b, model = detection_task.get_detection_objs(
            im, model,
            old_img_w,
            old_img_h,
            old_img_b
        )

        if algorithm_config.classify:
            detection_objs = apply_classifier(detection_objs, classifier, im, im0s)

        for i, detection in enumerate(detection_objs):
            seen_obj += 1

            if input_config.webcam_enable:
                im0 = im0s[i].copy()
                source_path = Path(source_paths[i])
            else:
                im0 = im0s.copy()
                source_path = source_paths

            try:
                current_frames[i] = im0
            except IndexError:
                pass

            gain = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            box_annotator = Annotator(
                im0,
                line_width=output_video_config.line_thickness,
                example=str(class_names)
            )

            try:
                tracking_task.motion_compensation(i, current_frames[i], prev_frames[i])

                if detection is None or not len(detection):
                    continue

                detection[:, :4] = scale_coords(
                    im.shape[2:],
                    detection[:, :4],
                    im0.shape).round()

                results[i] = tracking_task.tracker_hosts[i].update(detection.cpu(), im0)

                if len(results[i]) < 0:
                    continue
            except (TypeError, IndexError):
                pass

            for result in results[i]:
                v_id = result[4]
                class_name = class_names[int(result[5])]
                confidence = result[6]
                bbox = result[0:4]

                # New vehicle being tracked
                if v_id not in vehicles.keys():
                    vehicles[v_id] = TrackedObject(
                            v_id,
                            class_name,
                            confidence,
                            bbox
                         )
                else:
                    vehicles[v_id].update(class_name, confidence, bbox)

                box_annotator.box_label(
                    vehicles[v_id].bbox,
                    vehicles[v_id].get_label(output_video_config),
                    color=colors(1, True)
                )

            log_msg += f'Frame {frame_index}: vehicles = {len(vehicles)}\n'

            if frame_index != 0 and frame_index % 300 == 0:
                asyncio.run(upload.main(vehicles.copy()))
                # vehicles = {}

            stream_result(box_annotator, im0, source_path, streaming_windows)

            prev_frames[i] = current_frames[i]


if __name__ == '__main__':
    with torch.no_grad():
        CLI(main, as_positional=False)
