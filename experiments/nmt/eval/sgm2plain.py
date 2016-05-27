#!/usr/bin/python
import os
import sys

run = os.system

def extract(line):
    p1 = line.find(">")
    p2 = line.rfind("<")
    return line[p1+1:p2].strip()

def sgm2plain(src_plain, trg_sgm):
    "Converse sgm format to plain format"
    fin_src_plain = file(src_plain , "r")
    fout = file(trg_sgm, "w")
    
    #head
    for line in fin_src_plain:
        if line.startswith('<seg'):
            print >> fout, extract(line)

if __name__ == "__main__" :
    if (len(sys.argv) != 3) :
        print >> sys.stderr, "exe in_plain out_sgm"
        sys.exit(0)

    src_plain = sys.argv[1]
    trg_sgm = sys.argv[2]

    sgm2plain(src_plain, trg_sgm)

