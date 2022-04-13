from multiprocessing import Pool

import numpy as np
import pandas as pd
# multiprocessing
# try using concurrent library here sometime
class DfParallel():
    def _init_(self,nchunks=50):
        self.nchunks = nchunks
        pass

    def paralelize(self,df,split_by,job):
        new_df = pd.DataFrame()
        inputs = self.get_inputs(df,split_by)
        outputs = self.do_job(job, inputs)
        for output in outputs:
            new_df = new_df.append(output,ignore_index=True)
        return new_df#.sort_values(split_by).reset_index()

    def get_inputs(self,df,split_by):
        inputs = []
        for chunk in np.array_split((df[split_by].unique()), self.nchunks):
            inputs.append(df[df[split_by].isin(chunk)])
        return inputs

    def do_job(self,job, inputs):
        pool = Pool()
        outputs = pool.map(job, inputs)
        pool.close()
        pool.join()
        return outputs