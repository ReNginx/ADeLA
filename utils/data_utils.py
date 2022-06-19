import os


class DataUtils:
    @staticmethod
    def get_video_length(folder):
        content = os.listdir(folder)
        return len(content) // 3

    @staticmethod
    def apply_trans(obj, func):
        if type(obj) in (list, tuple):
            return [DataUtils.apply_trans(itm, func) for itm in obj]
        else:
            return func(obj)
