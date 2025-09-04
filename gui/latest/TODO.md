- Adjusting the FPS results in this error:
    Exception in thread Thread-1 (process_image):
    Traceback (most recent call last):
    File "C:\Users\Admin\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1009, in _bootstrap_inner
        self.run()
    File "C:\Users\Admin\AppData\Local\Programs\Python\Python310\lib\threading.py", line 946, in run
        self._target(*self._args, **self._kwargs)
    File "c:\Users\Admin\Documents\products\ASCII-Art-Studio\gui\latest\ascii_converter_gui.py", line 266, in process_image
        ascii_art = image_to_ascii(image, self.width_var.get(
    File "c:\Users\Admin\Documents\products\ASCII-Art-Studio\gui\latest\ascii_converter_gui.py", line 57, in image_to_ascii
        ascii_str += chars[pixel // (256 // len(chars))]
    IndexError: string index out of range

- Switching to detailed causes this error:
    Exception in thread Thread-1 (process_image):
    Traceback (most recent call last):
    File "C:\Users\Admin\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1009, in _bootstrap_inner
        self.run()
    File "C:\Users\Admin\AppData\Local\Programs\Python\Python310\lib\threading.py", line 946, in run
        self._target(*self._args, **self._kwargs)
    File "c:\Users\Admin\Documents\products\ASCII-Art-Studio\gui\latest\ascii_converter_gui.py", line 266, in process_image
        ascii_art = image_to_ascii(image, self.width_var.get(
    File "c:\Users\Admin\Documents\products\ASCII-Art-Studio\gui\latest\ascii_converter_gui.py", line 57, in image_to_ascii
        ascii_str += chars[pixel // (256 // len(chars))]
    IndexError: string index out of range

- Some images also say out of range
    ascii_str += chars[pixel // (256 // len(chars))]
    IndexError: string index out of range

- Add proper error handling to avoid crashing