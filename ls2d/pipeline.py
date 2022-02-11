###############################################################################
# Author: Bernard Hernandez
# Filename:
# Date:
# Description:
#
###############################################################################

# Generic
import uuid
import json



# Specific
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

# Specific
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# Configure json-pickle
jsonpickle.set_preferred_backend('json')
jsonpickle_numpy.register_handlers()

# PySML libraries
from ls2d.utils import _dump_pickle
from ls2d.utils import _dump_jsonpickle

class PipelineMemory(Pipeline):
    """This is an extension of the scikits pipeline.

    The only aim of this extension is to allow to save each
    of the models in memory just after they have been fit. To
    do that it uses a global variable to track what algorithm
    is being tested and what split is in.

    .. note: It can save the models in either json-pickle or
             pickle. Pickle might have some issues when running
             different numpy versions.

    .. note: Add option to chose whether to save as pickle.
             There are issues with the imputers...

    """

    def __init__(self, steps, memory_path=None,
                              memory_mode='pickle',
                              **kwargs):
        """The constructor

        .. note: It could be done with named_steps.

        Parameters
        ----------
        memory_path: str
            The path to store all the pipelines. Note that pipelines are
            all saved using the memory_mode specified. If the path does
            not exist it will be created.

        memory_mode: str (default: pickle)
            The format to store all the pipelines. The possible values
            are pickle or json-pickle.

        Returns
        """
        # Check memory mode
        _supported_memory_modes = ['pickle', 'jsonpickle']
        if memory_mode not in _supported_memory_modes:
            raise ValueError("""The value <%s> for memory mode is not supported.
                                Please choose from: %s""" % (memory_mode,
                                str(_supported_memory_modes)))

        # Set basic information
        self.memory_path = str(memory_path)
        self.memory_mode = memory_mode
        self.uuid = str(uuid.uuid4())
        #uuid.uuid1().int >> 64
        self.timestamp = str(datetime.utcnow())

        # Show id and timestamp
        #print(self.uuid, self.timestamp)

        # Set additional information
        self.slug_short = \
            '-'.join([s for s,l in steps])
        self.slug_long = \
            '-'.join([l.__class__.__name__ for s,l in steps])

        # Super constructor
        super(PipelineMemory, self).__init__(steps, **kwargs)


    #@property
    #def slug(self):
    #    """.. todo: deprecate"""
    #    return '-'.join([k for k in self.named_steps.keys()])

    #@property
    #def slugd(self):
    #    """.. todo: deprecate"""
    #    return '-'.join([v.__class__.__name__ for v in self.named_steps.values()])

    @property
    def filepath(self):
        """This method creates the filepath."""
        return '{0}/{1}/pipeline{2}/pipeline{2}-split{3}'.format( \
            self.memory_path, self.slug_short, self.pipeline, self.split)

    def memory_dump(self, filepath):
        """This method saves the pipeline."""
        if self.memory_mode == 'jsonpickle':
            _dump_jsonpickle('{0}.json'.format(filepath), self)
        elif self.memory_mode == 'pickle':
            _dump_pickle('{0}.p'.format(filepath), self)

    def fit(self, *args, **kwargs):
        """This method fits the model.

        Parameters
        ----------

        Returns
        --------
        """

        def __get_signature_hash(signature):
            """This method adds a signature to globals
            """
            # Create global hash of signatures
            if not 'signatures' in globals():
                globals()['signatures'] = {}

            # Add signature
            if signature not in globals()['signatures']:
                globals()['signatures'][signature] = \
                    [len(globals()['signatures']), 1]
            else:
                globals()['signatures'][signature][1] += 1

            # Return
            return globals()['signatures'][signature]

        # Fit
        r = super().fit(*args, **kwargs)

        # Find signature information
        self.pipeline, self.split = \
            __get_signature_hash(str(self.get_params()))

        # Create folder
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

        #self.named_steps['keras'][1].model = None

        # Dump pipeline in memory
        self.memory_dump(self.filepath)

        # Return
        return r