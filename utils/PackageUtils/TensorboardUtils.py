import cv2


def tensorboard_log_image_from_file(file_name, tb_writer, tb_title):
    plt2d_img = cv2.imread(file_name, 1)
    plt2d_img = cv2.cvtColor(plt2d_img, cv2.COLOR_BGR2RGB)
    tb_writer.add_image(tb_title, plt2d_img.transpose((2, 0, 1)))
