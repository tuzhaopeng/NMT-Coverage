#!/usr/bin/env python

import os, sys, time

def get_saved_batch(nohup_file):
    lines = open(nohup_file).readlines()
    lines.reverse()

    for i in xrange(len(lines)):
        if lines[i].startswith('Model saved,'):
            for j in xrange(i+1, len(lines)):
                if lines[j].startswith('.. iter '):
                    end = lines[j].find(' cost')
                    batch = int(lines[j][8:end].strip())
                    return batch

    if len(lines) > 0 and lines[-1].strip().endswith('Closing remaining open files:'):
        return -1

    return 0


def get_last_save(result_folder):
    folders = [folder for folder in os.listdir(result_folder) if folder.endswith('K')]
    saves = [float(f[:-1]) for f in folders]
    saves.sort(reverse=True)

    return int(saves[0])


def eval(folder):
    os.system('mkdir -p %s' % folder)

    os.system('''THEANO_FLAGS="on_unused_input=ignore,device=gpu0,floatX=float32" python ./sample.py --beam-search --beam-size 10 --verbose --source=./eval/dev_src.txt --trans %s/dev_tran.txt --state %s/search_state.pkl %s/search_model.npz 1>%s/dev.log.txt 2>%s/dev.err.txt''' % (folder, folder, folder, folder, folder))

    os.system('./eval/plain2sgm.py %s/dev_tran.txt eval/dev_src.sgm %s/dev_tran.sgm' % (folder, folder))
    os.system('./eval/mteval-v11b.pl -r eval/dev_ref.sgm -s eval/dev_src.sgm -t %s/dev_tran.sgm > %s/dev_tran.bleu' % (folder, folder))



if len(sys.argv) != 4:
    print "cmd result_folder interval(K batches) start(K batches)"
    sys.exit(1)

result_folder = sys.argv[1]
interval = int(float(sys.argv[2])*1000)
start = int(sys.argv[3])*1000

try:
    last_save = int(get_last_save(result_folder)*1000/interval)
except:
    last_save = 0

print 'last save:', last_save

while 1:
    saved_batch = get_saved_batch('nohup.out')
    print 'saved_batch:', saved_batch
    if saved_batch == -1:
        break

    if saved_batch >= start and saved_batch/interval > last_save:
        output_folder = '%s/%.0fK' % (result_folder, float(saved_batch)/1000)
        os.system('mkdir -p %s; cp search_* %s' % (output_folder, output_folder))
        eval(output_folder)
        last_save = saved_batch/interval

    time.sleep(60)
