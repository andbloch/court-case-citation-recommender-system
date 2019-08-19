import progressbar


class TrainingProgressBar:

    def __init__(self, num_epochs, num_batches):

        self.num_epochs = num_epochs
        self.num_batches = num_batches

        self.last_epoch = 1
        self.last_global_step = 0
        self.last_training_loss = float('inf')
        self.last_validation_loss = float('inf')
        self.last_validation_hit_rate = 0
        self.last_validation_ndcg = 0
        self.last_validation_hm = 0
        self.last_checkpoint = 0
        self.last_signature = ''

        self.pb_text_1 = None
        self.pb_text_2 = None
        self.bar = None

    def get_legend_string(self):
        s = \
        'GS: \tnumber of gradient steps ' + \
        'TL: \ttraining loss ' + \
        'VL: \tvalidation loss ' + \
        'HR: \tvalidation hit rate ' + \
        'NDCG: \tvalidation ndcg ' + \
        'CP: \tlast checkpoint ' + \
        'ETA: \testimated time of arrival (completion)'

    def start(self):

        pb_text_1_format = '[Epoch %(epoch)d/%(num_epochs)d, ' + \
                           'GS: %(steps)d, ' + \
                           'TL: %(train_loss).4f, ' + \
                           'VL: %(valid_loss).4f, ' + \
                           'HR: %(valid_hit_rate).2f%%, ' + \
                           'NDCG: %(valid_ndcg).2f%%, ' + \
                           'HM: %(valid_hm).2f%%, ' + \
                           'CP: %(checkpoint)d,'

        pb_text_1_dict = dict(
            num_epochs=self.num_epochs,
            epoch=self.last_epoch,
            steps=self.last_global_step,
            train_loss=self.last_training_loss,
            valid_loss=self.last_validation_loss,
            valid_hit_rate=self.last_validation_hit_rate,
            valid_ndcg=self.last_validation_ndcg,
            valid_hm=self.last_validation_hm,
            checkpoint=self.last_checkpoint
        )

        self.pb_text_1 = progressbar.FormatCustomText(pb_text_1_format,
                                                      pb_text_1_dict)

        widgets = [self.pb_text_1, ' ',
                   progressbar.ETA(), ' ',
                   progressbar.Percentage(), ' ',
                   progressbar.Bar()]

        self.bar = progressbar.ProgressBar(max_value=self.num_batches,
                                           widgets=widgets)
        self.bar.term_width = 160

    def update_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs
        self.pb_text_1.update_mapping(num_epochs=num_epochs)

    def update_epoch(self, epoch):
        self.last_epoch = epoch
        self.pb_text_1.update_mapping(epoch=epoch)

    def update_steps(self, steps):
        self.last_global_step = steps
        self.pb_text_1.update_mapping(steps=steps)

    def update_training_loss(self, training_loss):
        self.last_training_loss = training_loss
        self.pb_text_1.update_mapping(train_loss=training_loss)

    def update_validation_loss(self, validation_loss):
        self.last_validation_loss = validation_loss
        self.pb_text_1.update_mapping(valid_loss=validation_loss)

    def update_validation_hit_rate(self, validation_hit_rate):
        self.last_validation_hit_rate = validation_hit_rate
        self.pb_text_1.update_mapping(valid_hit_rate=validation_hit_rate)

    def update_validation_ndcg(self, validation_ndcg):
        self.last_validation_ndcg = validation_ndcg
        self.pb_text_1.update_mapping(valid_ndcg=validation_ndcg)

    def update_validation_hm(self, validation_hm):
        self.last_validation_hm = validation_hm
        self.pb_text_1.update_mapping(valid_hm=validation_hm)

    def update_last_checkpoint(self, last_checkpoint_timestep):
        self.last_checkpoint = last_checkpoint_timestep
        self.pb_text_1.update_mapping(checkpoint=last_checkpoint_timestep)

    def update_batch_counter(self, i):
        self.bar.update(i)

    def finish(self):
        self.bar.finish()