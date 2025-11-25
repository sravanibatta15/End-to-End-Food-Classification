import numpy as np
import cv2
import sys
from log_code import setup_logging
logger=setup_logging('common')

class Model:
    def model(model_obj, img_path):
        try:
            logger.info("Started prediction...")
            image = cv2.imread(img_path, 1)
            re_sizedimage = cv2.resize(image, (256, 256))
            input_image = np.expand_dims(re_sizedimage, axis=0)
            y_p = model_obj.predict(input_image)
            return int(np.argmax(y_p))
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            logger.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_value}")