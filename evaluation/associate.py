#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import numpy

def read_file_list(filename, remove_bounds):
    """
    读取文本文件中的数据列表。
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    
    # 修复：如果需要剔除边界数据
    if remove_bounds:
        lines = lines[100:-100]
        
    # 解析行数据
    content_list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    # 格式化为 (timestamp, [data...])
    formatted_list = [(float(l[0]), l[1:]) for l in content_list if len(l)>1]
    return dict(formatted_list)

def associate(first_list, second_list, offset, max_difference):
    """
    根据时间戳关联两个字典。
    """
    # 修复：Python 3 中 .keys() 返回的是视图，需要转成 list 才能进行 remove 操作
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    
    # 按时间差从大到小排序，确保最小差值的优先匹配
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将两个带时间戳的数据文件进行关联（例如 RGB 和 Depth）')
    parser.add_argument('first_file', help='第一个文本文件 (格式: timestamp data)')
    parser.add_argument('second_file', help='第二个文本文件 (格式: timestamp data)')
    parser.add_argument('--first_only', help='仅输出第一个文件的匹配行', action='store_true')
    parser.add_argument('--offset', help='第二个文件时间戳的偏移量 (默认: 0.0)', default=0.0)
    parser.add_argument('--max_difference', help='允许的最大时间差 (默认: 0.02s)', default=0.02)
    args = parser.parse_args()

    # 修复：传入缺失的 remove_bounds 参数
    first_list = read_file_list(args.first_file, False)
    second_list = read_file_list(args.second_file, False)

    matches = associate(first_list, second_list, float(args.offset), float(args.max_difference))    

    # 打印输出结果
    if args.first_only:
        for a, b in matches:
            print("%f %s" % (a, " ".join(first_list[a])))
    else:
        for a, b in matches:
            # 格式：rgb_timestamp rgb_path depth_timestamp depth_path
            print("%f %s %f %s" % (a, " ".join(first_list[a]), b - float(args.offset), " ".join(second_list[b])))
