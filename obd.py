import cv2,os,sys
import numpy as np
import datetime
import argparse
import tensorflow as tf
from functions import box_details,draw_bbox
from PIL import Image

interpreter = tf.lite.Interpreter(model_path='models/yolov4-416.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def main(args):
    image_path=args.input_path#'/Users/rahulbadhan/Desktop/Sorting/data/frame330.jpg'
    input_size=416
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = []
    for i in range(1):
        images_data.append(image_data)

    images_data = np.asarray(images_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = box_details(pred[0], pred[1], score_threshold=arg.thr,input_shape=tf.constant([input_size, input_size]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.25)

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    image=draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('output/output.jpg', image)
    print('**************** Image Write Successfully ****************')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OD')
    parser.add_argument('--input_path', type=str, default='/')
    parser.add_argument('--thr', type=float, default=0.25)
    # parser.add_argument('--output_path', type=str, default='/')
    arg = parser.parse_args()
    main(arg)








