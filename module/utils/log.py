import os
import datetime
import time

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        self.time0 = None
        self.epoch0 = 0
        if os.path.exists(log_path):
            os.remove(log_path)

    def write(self, msg, do_print=True):
        if do_print:
            print(msg)  # print the message 
        with open(self.log_path, "a") as log_file:
            log_file.write('{}\n'.format(msg))  # save the message

    def format_seconds(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return h, m, s

    def print_current_losses(self, epoch, n_epochs, losses):
        now = datetime.datetime.now()
        timestamp = "{}-{:02}-{:02} {:02}:{:02}:{:02}s".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

        if self.time0 is None:
            self.time0 = time.time()
            self.epoch0 = epoch
            d_t = 0
            d_epoch = 1
        else:
            d_t = time.time() - self.time0
            d_epoch = epoch - self.epoch0

        h, m, s = self.format_seconds(d_t)
        message = '[{}] ({:02.0f}:{:02.0f}:{:02.0f}) {:.2f}s/epoch. Epoch: {} | '.format(timestamp, h, m, s, d_t/(d_epoch+1e-6), epoch)
        for k, v in losses.items():
            if k.lower() == 'total':
                continue
            message += '%s:%.7f ' % (k, v)
        self.write(message)
