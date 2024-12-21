#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : test.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl)
@Date   : 12/12/2024
@Time   : 20:21:33
@Info   : Description of the script
"""
from config import get_config

def main():
   parser = get_config().parse_args()
   parser.agents_num
   
   

if __name__ == '__main__':
    main()