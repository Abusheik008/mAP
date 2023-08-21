from PIL import Image


class Reusable_code():

  def get_image_size(image_path):
        try:
            img = Image.open(image_path)
            width, height = img.size
            return width, height
        except(IOError, OSError):
            return None,None

  def get_image_format(image_path):
      try:
        for image_file in os.listdir(image_path):
            if image_file.endswith('.png'):
              img_format = ".png"
              return img_format
            elif image_file.endswith('.jpg'):
              img_format = ".jpg"
              return img_format
            elif image_file.endswith('.jpeg'):
              img_format = ".jpeg"
              return img_format
      except Exception as e:
        print(f"There is some error with image path : {str(e)}")




