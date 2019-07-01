import glob
import os
import shutil
from typing import List


class FileUtils(object):

    @staticmethod
    def read_lines(file_path: str, to_lower: bool = False) -> List[str]:
        a_list = []
        f = open(file_path, "r", newline='', encoding="utf-8")
        for line in f:
            line = line.rstrip("\n\r")
            if to_lower:
                line = line.lower()

            a_list.append(line)
        f.close()

        return a_list

    @staticmethod
    def read_directory(directory_path: str) -> List[str]:
        return glob.glob(directory_path + '**/*.*', recursive=True)

    @staticmethod
    def read_txt_file(file_path: str) -> str:
        with open(file_path, 'r', encoding="utf-8") as file:
            text_content = file.read()
        return text_content

    @staticmethod
    def get_abs_path(base_path, relative_path):
        return os.path.join(os.path.dirname(os.path.realpath(base_path)), relative_path)

    @staticmethod
    def delete_files(path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    @staticmethod
    def mkdir(dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    @staticmethod
    def save_file(file, dir_name, filename) -> str:
        FileUtils.mkdir(dir_name)
        upload_path = os.path.join(dir_name, filename)
        file.save(upload_path)
        return upload_path
