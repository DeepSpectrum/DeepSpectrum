import tensorflow as tf
from os.path import join, basename, dirname
from os import makedirs,listdir,getcwd,remove
import shutil
import re

class BestCheckpointExporter(tf.estimator.Exporter):
        def __init__(self, name, minimize=True, metric='average_loss'):
            self._name = name
            self._exportfolder = join('export',self._name)
            self._minimize = minimize
            self._best_value = None
            self._best_value_step = None
            self._metric = metric

        @property
        def name(self):
            return self._name
        def export(self, *args, **kwargs):
            eval_loss = kwargs['eval_result'][self._metric]
            global_step = kwargs['eval_result']["global_step"]
            checkpoint_path = kwargs['checkpoint_path']
            export_path = kwargs['export_path']
            del args, kwargs
            print("| EVAL > Global step {}: aver_loss = {}. Best so far: {} (step {})".format(global_step,eval_loss,self._best_value,self._best_value_step))
            if self._best_value is None:
                makedirs(join(getcwd(),export_path))
            if ( self._best_value is None or
                 (self._minimize and (eval_loss < self._best_value)) or
                 (not self._minimize and (eval_loss > self._best_value))):
                    self._best_value = eval_loss
                    self._best_value_step = global_step
                    # Remove previous best model
                    for f in listdir(export_path):
                        if re.search("model.ckpt", f):
                            remove(join(export_path, f))
                    # Copy checkpoint to best checkpoint folder (replace old)
                    model_files = [f for f in listdir(join(getcwd(),dirname(checkpoint_path))) if re.match(basename(checkpoint_path)+'.*', f)]
                    for f in model_files:
                        # shutil.copy(join(dirname(checkpoint_path),f),
                        #             join(export_path,f.replace(basename(checkpoint_path),'model.ckpt-best')))
                        shutil.copy(join(dirname(checkpoint_path),f),
                                    join(export_path,f))
                        shutil.copy(join(dirname(checkpoint_path),'params.pickle'),
                                    join(export_path,'params.pickle'))
                    # Save name of new best model
                    outF = open(join(export_path,"checkpoint"), "w")
                    outF.write("model_checkpoint_path: \"" + basename(checkpoint_path) + "\"")
                    outF.close()
                    print("       * NEW BEST")
