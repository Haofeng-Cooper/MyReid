# -*- coding: utf-8 -*-

import os
from queue import Queue, LifoQueue


FLAG_FILE = "-f"
FLAG_FOLDER = "-d"
FLAG_UNDEFINED = "--"


class FileFolder():
    def __init__(self, path, father=None, FILE_FLAG=None):
        if father is not None and not isinstance(father, FileFolder):
            raise TypeError("类型错误！请检查所给参数的类型是否符合要求！")

        # 文件或文件夹的绝对路径
        self.path = os.path.abspath(path)
        # 文件或文件夹名字
        self.name = os.path.basename(self.path)
        # 文件或文件夹的类型。普通文件为'-f'，文件夹为'-d'，其他为'--'
        self.type = FILE_FLAG if FILE_FLAG else check_file_type(self.path)
        # 包含的直接文件列表，有序列表
        self.child_files = []
        # 包含的直接子文件夹列表，有序列表
        self.child_folders = []

        # 所属上层目录。FileFolder类型
        self.father = None
        # 在所属文件夹下的排列顺序。文件和文件夹分开排序。文件夹从负数开始排，最后一个文件夹为-1。
        # 文件从正数开始排，第一个文件为1
        self.order = 0
        # 文件或文件夹的层级。root为0级，每向内一层增加一级
        self.level = 0

        # 其余参数设定
        if isinstance(father, FileFolder):
            self.father = father
            if self.type == FLAG_FILE:
                self.order = father.child_files.index(self.path)
            elif self.type == FLAG_FOLDER:
                self.order = father.child_folders.index(self.path)
            else:
                self.order = -1
            self.level = father.level + 1
        self.child_files = list_child_files(self)
        self.child_folders = list_child_folders(self)


    def __str__(self):
        folder_names = ""
        for folder in self.child_folders:
            folder_names += folder +"\n"
        file_names = ""
        for file in self.child_files:
            file_names += file + "\n"
        return "文件或文件夹属性：\n{}\t{}\t<{}>\t{}\t{}\n\n" \
               "所含子文件夹：\n{}\n" \
               "所含子文件：\n{}" \
               "----------------------------------------------------------------------------------------------".format(
            self.type, self.order, self.name, self.level, self.father, folder_names, file_names)



def list_child_files(fileFolder):
    path = os.path.abspath(fileFolder.path)
    result = []
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                result.append(file_path)
    return sorted(result)

def list_child_folders(fileFolder):
    path = os.path.abspath(fileFolder.path)
    result = []
    if os.path.isdir(path):
        files = os.listdir(path)
        for folder in files:
            file_path = os.path.join(path, folder)
            if os.path.isdir(file_path):
                result.append(file_path)
    return sorted(result)

def check_file_type(path):
    if os.path.isfile(path):
        return FLAG_FILE
    elif os.path.isdir(path):
        return FLAG_FOLDER
    else:
        return FLAG_UNDEFINED

def check_class(x, A_tuple):
    if isinstance(x, A_tuple):
        return True
    else:
        raise TypeError("类型错误！请检查所给参数的类型是否符合要求！")

def cmp_fileFolder(element):
    # return element.belongs
    return element.path

def is_wanted(fileFolder):
    if isinstance(fileFolder, FileFolder):
        if len(fileFolder.child_files) > 0 and len(fileFolder.child_folders) == 0:
            return True
    return False

def list_all_child_folders(root_dir_path):
    path = os.path.abspath(root_dir_path)

    folder_queue = LifoQueue()
    folder_list = []
    if os.path.isdir(path):
        root = FileFolder(path)
        folder_queue.put(root)
        while not folder_queue.empty():
            cur = folder_queue.get()
            folder_list.append(cur.path)
            size = len(cur.child_folders)

            for index in range(size):
                child_folder_path = cur.child_folders[size-1-index]
                child_folder = FileFolder(child_folder_path, father=cur, FILE_FLAG=FLAG_FOLDER)
                folder_queue.put(child_folder)

    return folder_list




if __name__ == "__main__":
    # print(FileFolder("D:/PycharmProjects/useOpenPose/my_op_lib"))
    # print(FileFolder("F:/super mario maker 2"))
    # result = list_all_child_folders("D:/PycharmProjects/useOpenPose/my_op_lib")
    result = list_all_child_folders("D:\\reid_datasets\\prid_2011\\multi_shot")
    for r in result:
        print(r)

