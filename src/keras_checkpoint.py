import json
import warnings
import numpy as np
import os
import keras

class KerasCheckpoint(keras.callbacks.Callback):

    """Save the model and stats after every epoch.
    `snapshot_path` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    # Arguments
        snapshot_path: string, path to save the model and stats file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, snapshots_path, label=None, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(KerasCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.snapshots_directory = snapshots_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.label = label
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.stats = {}
        if self.label:
            self.stats['label'] = self.label
        self.stats['type'] = 'hdf5'

        for k, v in logs.items():
            self.stats[k] = v

        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            snapshot_path = self.snapshots_directory + "/{epoch:02d}"
            snapshot_directory = snapshot_path.format(epoch=epoch, **logs)
            if not os.path.exists(snapshot_directory):
                os.makedirs(snapshot_directory)
            model_filepath = snapshot_directory + '/model.hdf5'
            stats_filepath = snapshot_directory + '/stats.json'
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, model_filepath))
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving stats to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, stats_filepath))
                        self.best = current
                        self.model.save_weights(model_filepath, overwrite=True)
                        with open(stats_filepath, 'wb') as f:
                            f.write(json.dumps(self.stats))
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, model_filepath))
                    print('Epoch %05d: saving stats to %s' % (epoch, stats_filepath))
                self.model.save_weights(model_filepath, overwrite=True)
                with open(stats_filepath, 'wb') as f:
                    f.write(json.dumps(self.stats))


