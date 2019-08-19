

def run_training(CONFIG):
    """
    DESCRIPTION

    Runs an entire training of a recommender model for a given configuration.
    The experiment's parameters can be parametrized via the CONFIG dictionary.


    CODING CONVENTION OF FILE

    All UPPERCASE variables/pointers remain constant (once they are set up) and
    are guaranteed to be in the scope of the training context. This facilitates
    the access across functions and avoids having to pass them around, or
    re-assign them unnecessarily. Accepting the side effects actually makes the
    code simpler compared to trying to pass around/shadow/re-assign variables to
    avoid side effects.
    """


    # IMPORTS ##################################################################


    # system and time
    import os
    import time
    import math

    # yaml configuration
    import yaml

    # GPU stats
    from GPUtil import getAvailable as get_available_GPUs

    # datasets
    from data import CitationDataset
    from data.collate_batch import collate_batch_examples

    # training progress
    from training import TrainingProgressBar

    # pytorch
    import torch
    import torch.nn as nn
    import torch.utils.data
    import torch.nn.functional as F
    from torch import autograd

    # progress bar
    from tqdm import tqdm

    # ranking models
    from ranking_models import ItemPopularity
    from ranking_models import NCF

    # optimizers
    from torch.optim import Adam

    # pytorch tensorboardX
    from tensorboardX import SummaryWriter

    # logger
    import logging

    # pretty printing of yaml
    from pprint import pformat

    # quintuple service (not needed, but set up to be kept alive in memory)
    from data.CaseQuintupleService import CaseQuintupleService


    # SET-UP OF QUINTUPLE SERVICE ##############################################


    # just set up quintuple service s.t. it is kept alive among all batch
    # loaders (to avoid multiple loadings and copies of case quintuples)
    Q = CaseQuintupleService()


    # ACCESSING OF CONFIG FILE #################################################


    def safe_dict_get(key, dict):
        return dict[key] if key in dict else None


    # CREATE RESULTS DICTIONARY ################################################


    # create results dictionary from CONFIG
    # (after here there will be no updates to the config file)
    RESULTS = CONFIG


    # LOGGING CONFIGURATION ####################################################


    LOGGING_FORMAT = '%(message)s'
    logger = logging.getLogger('training_loop')
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)

    # print config file
    logger.info('\n---Configuration---')
    logger.info(pformat(CONFIG))


    # DEVICE SETTINGS ##########################################################


    # device settings
    DEVICE = CONFIG['DEVICE']
    NUM_GPUS = CONFIG['NUM_GPUS']
    USED_GPU_NUMBERS = CONFIG['USED_GPU_NUMBERS']

    # print device configuration
    logger.info('\n---Used Devices---')

    # get list of available GPUs
    avaliable_GPUs = get_available_GPUs(limit=math.inf,
                                        maxLoad=0.01,
                                        maxMemory=0.1)

    # default to cuda device if not specified
    if DEVICE is None:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(DEVICE)

    # further cuda settings (restriction of used GPUs)
    if DEVICE.type == 'cuda':

        if USED_GPU_NUMBERS:
            # restrict visibility to predefined GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = USED_GPU_NUMBERS
            logger.info('Using predefined GPUs: ['+USED_GPU_NUMBERS+']')
        else:
            # throw an error if there are not enough GPUs available
            if NUM_GPUS is not None and len(avaliable_GPUs) < NUM_GPUS:
                raise RuntimeError('Not enough GPUs available!')

            # restrict visible GPUs to number of used GPUs
            used_GPUs_list = avaliable_GPUs
            if NUM_GPUS is not None:
                used_GPUs_list = avaliable_GPUs[:NUM_GPUS]

            # convert used GPUs to list of strings
            used_GPUs = ','.join(list(map(str, used_GPUs_list)))

            # restrict visibility to set of used GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = used_GPUs

            # print used device
            num_gpus_used = len(used_GPUs_list)
            logger.info('List of GPUs used: ['+used_GPUs+']   ' + \
                        '('+str(num_gpus_used)+' GPU' + \
                        ('s' if num_gpus_used > 1 else '') + ')')
    else:
        logger.info('Using CPU')


    # EXPERIMENT CREATION AND PARAMETRIZATION ##################################

    # tranining and batching settings
    NUM_WORKERS = CONFIG['NUM_WORKERS']
    NUM_EPOCHS = CONFIG['NUM_EPOCHS']

    # progress printing, saving and validation
    # (EVERY = x, means every x gradient steps (GS))
    PRINT_TRAINING_PROGRESS_EVERY = CONFIG['PRINT_TRAINING_PROGRESS_EVERY']
    VALIDATE_AND_CHECKPOINT_EVERY = CONFIG['VALIDATE_AND_CHECKPOINT_EVERY']
    PRINT_NUM_GRADIENT_STEPS_EVERY = CONFIG['PRINT_NUM_GRADIENT_STEPS_EVERY']

    # validation settings
    DO_VALIDATION = CONFIG['DO_VALIDATION']

    # data settings
    DATA = CitationDataset(CONFIG['DATA_OWNER'],
                           CONFIG['DATA_NAME'],
                           CONFIG['LOSS_NAME'],
                           CONFIG['NUM_NEGATIVE_EXAMPLES'],
                           CONFIG['HIT_RATE_NUM_NEGATIVE_EXAMPLES'],
                           CONFIG['NET_NAME'])

    # model settings
    NUM_FACTORS = safe_dict_get('NUM_FACTORS', CONFIG)
    NUM_LAYERS = safe_dict_get('NUM_LAYERS', CONFIG)
    WORD_EMBEDDING_DIM = safe_dict_get('WORD_EMBEDDING_DIM', CONFIG)
    GMF_SENT_EMBED_NUM_LAYERS = safe_dict_get('GMF_SENT_EMBED_NUM_LAYERS', CONFIG)
    GMF_SENT_EMBED_DIM = safe_dict_get('GMF_SENT_EMBED_DIM', CONFIG)
    GMF_DOC_EMBED_NUM_LAYERS = safe_dict_get('GMF_DOC_EMBED_NUM_LAYERS', CONFIG)
    MLP_SENT_EMBED_NUM_LAYERS = safe_dict_get('MLP_SENT_EMBED_NUM_LAYERS', CONFIG)
    MLP_SENT_EMBED_DIM = safe_dict_get('MLP_SENT_EMBED_DIM', CONFIG)
    MLP_DOC_EMBED_NUM_LAYERS = safe_dict_get('MLP_DOC_EMBED_NUM_LAYERS', CONFIG)

    # model instantiation and run name configuration
    NET = None
    NET_NAME = CONFIG['NET_NAME']
    if NET_NAME == 'ItemPopularity':
        NET = ItemPopularity(DATA.get_training_ranked_items(),
                             DATA.get_training_popularities())
    elif NET_NAME == 'NCF':
        NET = NCF(NUM_FACTORS,
                  NUM_LAYERS,
                  WORD_EMBEDDING_DIM,
                  GMF_SENT_EMBED_NUM_LAYERS,
                  GMF_SENT_EMBED_DIM,
                  GMF_DOC_EMBED_NUM_LAYERS,
                  MLP_SENT_EMBED_NUM_LAYERS,
                  MLP_SENT_EMBED_DIM,
                  MLP_DOC_EMBED_NUM_LAYERS)

    # deactivate training for item popularity model
    if isinstance(NET, ItemPopularity):
        NUM_EPOCHS = 0

    # regularization settings
    USE_REGULARIZATION = CONFIG['USE_REGULARIZATION']
    REGULARIZATION_LAMBDA = CONFIG['REGULARIZATION_LAMBDA']

    # summary settings
    SUMMARIZE_HISTOGRAMS = CONFIG['SUMMARIZE_HISTOGRAMS']


    # OPTIMIZER SETTINGS #######################################################


    # get optimizer parameters
    LEARNING_RATE = CONFIG['LEARNING_RATE']
    BETAS = safe_dict_get('BETAS', CONFIG)
    EPS = safe_dict_get('EPS', CONFIG)
    WEIGHT_DECAY = CONFIG['WEIGHT_DECAY']
    AMSGRAD = safe_dict_get('AMSGRAD', CONFIG)

    # create optimizers
    OPTIMIZER = Adam(
        NET.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD
    )


    # LOSS DEFINITION ##########################################################


    def bpr_loss(ranks):
        """
        :param ranks: the higher the rank of an item (on an absolute scale
        from -inf to +inf), the more relevant it is.
        :return:
        """

        # split ranks into positive and negative components
        xh_ui = ranks[:,0]      # ranks of positive items
        xh_uj = ranks[:,1]      # ranks of negative items

        # compute loss across batch
        # https://www.wolframalpha.com/input/?i=-loge(1%2F(1%2Be%5E(-x)))
        l = (-F.logsigmoid(xh_ui - xh_uj)).mean()

        return l


    # select loss function
    LOSS = bpr_loss

    # computation of loss
    def compute_loss(ranks):
        full_loss = None
        training_loss = LOSS(ranks)  # training loss computation
        regularization_loss = None
        if USE_REGULARIZATION:  # use regularization if desired
            regularization_loss = \
                REGULARIZATION_LAMBDA * NET_REF.get_regularization_loss()
            full_loss = training_loss + regularization_loss
        else:
            full_loss = training_loss
        return full_loss, training_loss, regularization_loss


    # GPU DATA-PARALLELIZATION #################################################


    # if there are more than one DEVICE, and we want to use GPUs use them all
    use_cuda = (DEVICE.type == 'cuda')
    want_many_GPUs = (NUM_GPUS is None or NUM_GPUS > 1)
    has_many_GPUs = (torch.cuda.device_count() > 1)
    if use_cuda and want_many_GPUs and has_many_GPUs:
        num_gpus = torch.cuda.device_count()
        logger.info('Creating data-parallel net using '+str(num_gpus)+' GPUs!')
        # creates the DATA-parallel model
        # For example: this will split a batch of 30 items as follows:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # so the data is split along the batch dimension
        # For more see:
        # https://medium.com/huggingface/training-larger-batches-practical-tips
        # -on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
        NET = nn.DataParallel(NET)


    # create reference to network to access non-default functions and properties
    NET_REF = NET.module if isinstance(NET, nn.DataParallel) else NET


    # BATCH GENERATORS #########################################################


    # note that the batch-generators need to be instantiated every time again in
    # order to produce DIFFERENT negative samples every time.

    # determine whether to pin memory
    pin_memory = True if DEVICE.type == 'cuda' else False

    # define collate function (partially applied with device)
    def device_collate_fn(batch):
        return collate_batch_examples(batch, DEVICE, NET_NAME)

    # create batch generator for training data
    TRAINING_BATCH_LOADER = None
    def create_training_batch_loader():
        return torch.utils.data.DataLoader(
            DATA.get_training_set(),
            batch_size=CONFIG['TRAIN_BATCH_SIZE'],
            shuffle=True,                   # shuffles data every epoch
            num_workers=NUM_WORKERS,        # #CPU workers batching to DEVICE
            pin_memory=pin_memory,          # copies tensors to CUDA pinned mem.
                                            # for faster transfers
            collate_fn=device_collate_fn    # function to collate a batch
        )

    # create batch generator for validation data on all items
    def create_validation_batch_loader():
        return torch.utils.data.DataLoader(
            DATA.get_validation_set(),
            batch_size=CONFIG['VALID_BATCH_SIZE'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            collate_fn=device_collate_fn
        )

    # create batch generator for validation data on subsampled sets of items
    def create_validation_subsampling_batch_loader():
        return torch.utils.data.DataLoader(
            DATA.get_subsampled_validation_set(),
            batch_size=CONFIG['VALID_BATCH_SIZE'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            collate_fn=device_collate_fn
        )

    # create batch generator for test data on subsampled sets of items
    def create_test_subsampling_batch_loader():
        return torch.utils.data.DataLoader(
            DATA.get_subsampled_test_set(),
            batch_size=CONFIG['TEST_BATCH_SIZE'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            collate_fn=device_collate_fn
        )


    # CHECKPOINTS AND SUMMARIES ################################################


    # determine run prefix if it exists
    run_prefix = ''
    supplied_run_prefix = safe_dict_get('RUN_PREFIX', CONFIG)
    if supplied_run_prefix:
        run_prefix = '/'+supplied_run_prefix

    # determine output directory for checkpoints and summaries
    timestamp = str(int(time.time()))
    run_comment = CONFIG['RUN_COMMENT']
    run_comment = run_comment if run_comment is not None else '_'
    OUT_DIR = os.path.abspath(os.path.join(os.path.curdir,
                                           'runs'+run_prefix,
                                           NET_REF.get_net_name(),
                                           NET_REF.get_run_name(),
                                           timestamp,
                                           run_comment))

    logger.info('\n---Summaries---')
    logger.info('Writing checkpoints and summaries to: {}'.format(OUT_DIR))

    # create checkpoint directory and file name
    checkpoint_dir = os.path.abspath(os.path.join(OUT_DIR, 'checkpoints'))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    CHECKPOINT_FILENAME = os.path.join(checkpoint_dir, 'checkpoint.tar')


    # checkpoint creation function
    def create_checkpoint(epoch, global_step, hit_rate, ndcg, hm,
                          validation_full_loss):
        checkpoint_file_path = CHECKPOINT_FILENAME
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'hit_rate': hit_rate,
            'ndcg': ndcg,
            'hm': hm,
            'validation_full_loss': validation_full_loss,
            'model_state_dict': NET.state_dict(),
            'optimizer_state_dict': OPTIMIZER.state_dict(),
        }, checkpoint_file_path)


    # create summary writer (will be used by functions below
    WRITER = SummaryWriter(OUT_DIR)


    def write_activations_summary(global_step):
        # TODO: add summarizing of activations (sparsity, norm, etc..)
        return 0


    def write_optimizer_summary(global_step):
        # TODO: write optimizer summary, think about what things should be
        # TODO: summarized (e.g., momentum, ...)
        return 0


    # function to write gradient summaries
    def write_gradient_summaries(global_step):
        for name, param in NET.named_parameters():
            if param.requires_grad:
                # compute gradient norm (normalized by number of parameters)
                norm = param.grad.data.norm(2).item()/param.numel()
                # compute sparsity of gradients
                # (how many percent of gradients are zero?)
                sparsity = 100.0 * \
                    (1.0-float(param.grad.data.nonzero().size(0))/param.numel())
                WRITER.add_scalar('grads/norm/'+name, norm,
                                  global_step=global_step)
                WRITER.add_scalar('grads/sparsity/'+name, sparsity,
                                  global_step=global_step)
                if SUMMARIZE_HISTOGRAMS:
                    WRITER.add_histogram('grads/hist/'+name, param.grad.data,
                                         global_step=global_step)


    # function to write loss summaries
    def write_training_summary(training_loss, global_step):
        WRITER.add_scalar('loss/train', training_loss, global_step=global_step)


    # write validation summaries
    def write_validation_summary(loss, hit_rate_subsampled, global_step):
        # create validation loss summary
        WRITER.add_scalar('loss/validation', loss, global_step=global_step)
        # create subsampled hit rate summary
        WRITER.add_scalar('hit_rate_subsampled/validation',
                          hit_rate_subsampled, global_step=global_step)


    # CONFIGURATION LOGGER #####################################################


    with open(os.path.join(OUT_DIR, 'CONFIG.yaml'), 'w') as outfile:
        yaml.dump(CONFIG, outfile, default_flow_style=False)


    # VALIDATION FUNCTIONS #####################################################


    # keep track of best validation hit rate and epoch
    BEST_VALIDATION_HR = \
        torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    BEST_VALIDATION_NDCG = \
        torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    BEST_VALIDATION_HM = \
        torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    BEST_VALIDATION_FULL_LOSS = \
        torch.tensor(100000000.0, dtype=torch.float32, device=DEVICE)
    BEST_VALIDATION_EPOCH = -1


    def HR_NDCG_and_HM_at_k_subsampled(k=10, mode='validation'):
        """
        Computes the HR@k and NDCG@k of the model where the predictions are done
        for N-1 uniformly at random sampled items which have not been interacted
        with yet and the target item (so in total N items).
        :param k: the number of top-k elements to compare for a hit.
        :param mode: 'validation' or 'test'
        :return: the HR@k and NDCG@k and HM@k where the number of negative samples is
        parametrized in the batch loaders
        """

        # create tensors to accumulate hit rate and normalized discounted
        # cumulative gain
        hit_rate = torch.empty(1, device=DEVICE, dtype=torch.float32)
        hit_rate[0] = 0.0

        ndcg = torch.empty(1, device=DEVICE, dtype=torch.float32)
        ndcg[0] = 0.0

        # create new validation batch (with new negative samples)
        batch_loader = None
        if mode == 'validation':
            batch_loader = create_validation_subsampling_batch_loader()
        elif mode == 'test':
            batch_loader = create_test_subsampling_batch_loader()

        # create validation_set_size tensor
        validation_set_size = torch.empty(1, device=DEVICE,
                                             dtype=torch.float32)
        # determine size of validation set and batch loader
        validation_set_size[0] = len(batch_loader.dataset)

        # for each validation example
        # the validation examples are rows of the form:
        # [user_id, pos_item_id, neg_item_id_(1), ..., neg_item_id_(n_samp.-1)]
        for validation_batch in batch_loader:

            # transfer validation batch to device
            for examples in validation_batch:
                examples[0] = examples[0].to(DEVICE)
                # Item Popularity performance optimiztion
                if NET_NAME != 'ItemPopularity':
                    examples[1] = examples[1].to(DEVICE)

            # get user and item idxs
            citing = validation_batch[0]
            cited = validation_batch[1:]

            # predict ranks for users and their item sets
            # the ranks contain rows of the form
            # [rank(user_id, pos_item_id), rank(user_id, neg_item_id_(1), ...]
            ranks = NET(citing, cited)

            # determine indices of top k rankings (for each row)
            top_k_rankings, top_k_corresponding_indices_of_rankings \
                = torch.topk(ranks, k=k, dim=1)

            # count number of rows which contain index 0. in other words:
            # check whether the positive example (which is at column index 0 in
            # the ranks tensor) was among the top k rankings
            num_hits = (top_k_corresponding_indices_of_rankings == 0).sum()

            # get the position of the target item among the top-k items
            # in other words: get the position at which index 0 is
            # from 0 to k-1
            indices_of_target_item_appearances = \
                (top_k_corresponding_indices_of_rankings == 0).nonzero()[:, 1]
            # add 2 to convert the indices to be the arguments to the log_2
            # discounting factor
            # (we add +1 because we want them to be in the range of 1 to k)
            # (we another +1 because that's how they are passed to log_2)
            indices_of_target_item_appearances += 2
            # convert the indices to floats to make sure they can be passed to
            # the log_2 function
            indices_of_target_item_appearances = \
                indices_of_target_item_appearances.type(torch.float32)
            ndcgs = (1.0 / torch.log2(indices_of_target_item_appearances)).sum()

            # add the number of hits to avg hit rate by averaging them
            hit_rate += (num_hits / validation_set_size).item()

            # add the ndcgs to the avg ndcg by averaging them
            ndcg += (ndcgs / validation_set_size).item()

        # adjust hit_rate and ndcg to percentage scale
        hit_rate = hit_rate * 100.0
        ndcg = ndcg * 100.0

        # compute the harmonic mean
        eps = torch.empty(1, device=DEVICE)
        eps[0] = 1e-10
        hm = 2.0 / ((1.0 / (hit_rate + eps)) + (1.0 / (ndcg + eps)))

        return hit_rate, ndcg, hm


    def get_validation_loss():
        """
        Computes the loss on the validation set. Note that this function cannot
        be simply combined for the training loss, as it uses a lot of other
        things like batching, etc.
        :return:
        """

        # create tensor to accumulate validation loss over batches
        validation_loss = torch.empty(1, device=DEVICE)
        validation_loss[0] = 0.0

        # create new validation batch (with new negative examples)
        batch_loader = create_validation_batch_loader()

        # determine validation set size
        validation_set_size = torch.zeros(1, device=DEVICE)
        validation_set_size[0] = len(batch_loader.dataset)

        # for each validation batch
        for validation_batch in batch_loader:

            # transfer batch data to DEVICE
            for item in validation_batch:
                item[0] = item[0].to(DEVICE)
                item[1] = item[1].to(DEVICE)

            # extract components in validation batch
            citing = validation_batch[0]        # citing document
            cited = validation_batch[1:]        # cited documents (pos, neg)

            # do predictions
            ranks = NET(citing, cited)

            # compute loss
            full_loss, training_loss, regularization_loss = compute_loss(ranks)

            # accumulate validation loss
            # (weighted average of averages weighted by batch sizes)
            batch_size = citing[0].shape[0]
            validation_loss += (batch_size / validation_set_size) * full_loss

        return validation_loss


    def validate_model():

        # deactivate gradient computation (also to gain performance)
        with torch.no_grad():

            # compute validation loss
            validation_loss = get_validation_loss()

            # compute subsampled validation hit-rate
            validation_hit_rate,\
            validation_ndcg, \
            validation_hm = \
                HR_NDCG_and_HM_at_k_subsampled(k=CONFIG['HIT_RATE_K'],
                                               mode='validation')

            return validation_loss, \
                   validation_hit_rate, \
                   validation_ndcg, \
                   validation_hm


    # EXECUTION TIME TRACKING ##################################################


    script_start_time = time.time()


    # TRAINING LOOP ############################################################

    MODEL_SELECTION = CONFIG['MODEL_SELECTION']

    # transfer the model to the device
    NET.to(DEVICE)

    # print list of parameters
    def print_list_of_parameters():
        logger.info('\n---Model Parameters---')
        logger.info('(*=trainable, /=not trainable)')
        num_params = 0
        for param in NET.named_parameters():
            trainable = ('*' if param[1].requires_grad else '/')
            run_name = str(param[0])
            dims = str(list(param[1].shape))
            logger.info(trainable+' '+run_name+': '+dims)
            if param[1].requires_grad:
                num_params += param[1].numel()
        logger.info('#trainable parameters: '+str(num_params))

    print_list_of_parameters()

    # notify starting of training loop
    if not isinstance(NET_REF, ItemPopularity):
        logger.info('\n---Starting Training Loop---')

    # initialize counter for number of gradient steps
    global_step = 0

    # create first training batch
    TRAINING_BATCH_LOADER = []
    if not isinstance(NET_REF, ItemPopularity):
        TRAINING_BATCH_LOADER = create_training_batch_loader()

    # initialize training progress bar
    num_training_batches = len(TRAINING_BATCH_LOADER)
    bar = TrainingProgressBar(NUM_EPOCHS, num_training_batches)

    # train for number of specified epochs
    epoch = 1
    while epoch <= NUM_EPOCHS:

        # initialize running average of training loss
        running_avg_training_loss = 0.0     # value of loss
        running_avg_training_loss_N = 0     # num. examples processed

        # start and create new training progress bar for each new epoch
        bar.start()
        bar.update_epoch(epoch)

        # ---Creation of new Batch Loader for Epoch-----------------------------

        # create a new training batch (with new negative samples)
        if epoch >= 2:
            TRAINING_BATCH_LOADER = create_training_batch_loader()

        # for each training batch
        for batch_num, train_batch in enumerate(TRAINING_BATCH_LOADER, 0):

            # TODO: activate this if you want to detect anomalies
            # with autograd.detect_anomaly():

            # ---Loading and Transferring of Batch to Device--------------------

            # transfer batch to device
            for examples in train_batch:
                examples[0] = examples[0].to(DEVICE)
                examples[1] = examples[1].to(DEVICE)

            # extract citing, cited_pos and cited_negs
            citing = train_batch[0]             # citing document
            cited = train_batch[1:]             # cited documents (pos, neg)

            # ---Zero-out Gradients---------------------------------------------

            # zero-out the parameter gradients
            OPTIMIZER.zero_grad()

            # ---Prediction-----------------------------------------------------

            # forward + backward + optimize
            ranks = NET(citing, cited)          # get ranks for pos, neg docs

            # ---Loss Computation ----------------------------------------------

            # TODO: plot these losses separately
            full_loss, training_loss, regularization_loss = compute_loss(ranks)

            # ---Gradient Step--------------------------------------------------

            full_loss.backward()                # compute grads through backprop
            OPTIMIZER.step()                    # do gradient step
            global_step += 1                    # count number of gradient steps

            # TODO: do this if you run again into memory problems!
            # https://discuss.pytorch.org/t/
            # best-practices-for-maximum-gpu-utilization/13863/6
            # del loss   # garbage collect loss to free mem

            # update number of gradient steps
            if global_step % PRINT_NUM_GRADIENT_STEPS_EVERY == 0:
                bar.update_steps(global_step)

            # ---Average Loss Accumulation--------------------------------------

            # accumulate average training loss
            # (weighted average of previous and current average loss)
            batch_size = citing[0].shape[0]
            new_N = running_avg_training_loss_N + batch_size
            scaling_factor_1 = running_avg_training_loss_N / new_N
            scaling_factor_2 = batch_size / new_N
            running_avg_training_loss = \
                scaling_factor_1*running_avg_training_loss +\
                scaling_factor_2*full_loss.item()
            running_avg_training_loss_N = new_N

            # ---Writing of Training Summaries----------------------------------

            # write training summaries
            # TODO: summarize training and regularization losses separately
            write_training_summary(full_loss.item(), global_step)
            write_gradient_summaries(global_step)
            write_optimizer_summary(global_step)

            # ---Printing of Training Progress----------------------------------

            # print training progress stats (num gradient steps, avg loss)
            if global_step % PRINT_TRAINING_PROGRESS_EVERY == 0:
                bar.update_training_loss(running_avg_training_loss)
                running_avg_training_loss = 0
                running_avg_training_loss_N = 0

            # ---Validation and Optional Checkpointing--------------------------

            # validate model if condition reached
            if global_step % VALIDATE_AND_CHECKPOINT_EVERY == 0 \
                and DO_VALIDATION:
                # run validation
                validation_loss, \
                validation_hit_rate, \
                validation_ndcg, \
                validation_hm = validate_model()

                # keep track of best validation hit rate and model
                do_checkpoint = False
                if validation_hit_rate > BEST_VALIDATION_HR:
                    BEST_VALIDATION_HR = validation_hit_rate
                    do_checkpoint = do_checkpoint or 'HR' in MODEL_SELECTION
                if validation_ndcg > BEST_VALIDATION_NDCG:
                    BEST_VALIDATION_NDCG = validation_ndcg
                    do_checkpoint = do_checkpoint or 'NDCG' in MODEL_SELECTION
                if validation_hm > BEST_VALIDATION_HM:
                    BEST_VALIDATION_HM = validation_hm
                    do_checkpoint = do_checkpoint or 'HM' in MODEL_SELECTION
                if validation_loss < BEST_VALIDATION_FULL_LOSS:
                    BEST_VALIDATION_FULL_LOSS = validation_loss
                    do_checkpoint = do_checkpoint or 'VL' in MODEL_SELECTION

                # do checkpoint (based on MODEL_SELECTION criteria)
                if do_checkpoint:
                    BEST_VALIDATION_EPOCH = epoch
                    # create checkpoint
                    create_checkpoint(epoch, global_step,
                                      validation_hit_rate,
                                      validation_ndcg,
                                      validation_hm,
                                      validation_loss)
                    bar.update_last_checkpoint(global_step)

                # write validation summary
                write_validation_summary(validation_loss,
                                         validation_hit_rate,
                                         global_step)

                # update progress bar with validation metrics
                bar.update_validation_loss(validation_loss)
                bar.update_validation_hit_rate(validation_hit_rate)
                bar.update_validation_ndcg(validation_ndcg)
                bar.update_validation_hm(validation_hm)

            # update progress bar with epoch progress
            bar.update_batch_counter(batch_num)

        # terminate progress bar at each epoch
        bar.finish()

    # free memory
    del TRAINING_BATCH_LOADER

    if not isinstance(NET, ItemPopularity):
        logger.info('Finished Training')


    # RELOAD BEST VALIDATION MODEL #############################################


    best_validation_hit_rate = None
    best_validation_ndcg = None
    best_validation_hm = None
    best_validation_loss = None
    best_validation_epoch = None
    if isinstance(NET, ItemPopularity) or not DO_VALIDATION:
        best_validation_epoch = -1
    else:
        logger.info('\n---Best Validation Model (Model Selection: ' + \
                    MODEL_SELECTION+')---')

        # reload checkpoint of model with best validation score
        checkpoint = torch.load(CHECKPOINT_FILENAME)
        # reload network
        NET.load_state_dict(checkpoint['model_state_dict'])
        # reload best validation score
        best_validation_hit_rate = checkpoint['hit_rate']
        # reload best ndcg
        best_validation_ndcg = checkpoint['ndcg']
        # reload best hm
        best_validation_hm = checkpoint['hm']
        # reload best validation loss
        best_validation_loss = checkpoint['validation_loss']
        # reload best epoch
        best_validation_epoch = checkpoint['epoch']

        # print model stats
        logger.info('HR@k:   \t%.2f%%' % best_validation_hit_rate)
        logger.info('NDCG@k: \t%.2f%%' % best_validation_ndcg)
        logger.info('HM@k:   \t%.2f%%' % best_validation_hm)
        logger.info('VL:     \t' + str(best_validation_loss.item()))
        logger.info('At Epoch %d/%d' % (best_validation_epoch, NUM_EPOCHS))


    # TEST SET EVALUATION ######################################################


    logger.info('\n---Computing Test Scores---')

    # get number of test evaluations
    NUM_TEST_EVALUATIONS = CONFIG['NUM_TEST_EVALUATIONS']

    # create array to store hit_rates
    test_hit_rates = torch.empty(NUM_TEST_EVALUATIONS, device=DEVICE)
    test_ndcgs = torch.empty(NUM_TEST_EVALUATIONS, device=DEVICE)
    test_hms = torch.empty(NUM_TEST_EVALUATIONS, device=DEVICE)

    # compute test HR@k TEST_NUM_EVALUATIONS times
    with torch.no_grad():
        for i in tqdm(range(NUM_TEST_EVALUATIONS)):
            test_hit_rates[i], \
            test_ndcgs[i], \
            test_hms[i] = \
                HR_NDCG_and_HM_at_k_subsampled(k=CONFIG['HIT_RATE_K'],
                                               mode='test')

    # compute mean and standard dev
    mean_test_hit_rate = test_hit_rates.mean()
    std_dev_test_hit_rate = test_hit_rates.std()
    mean_test_ndcg = test_ndcgs.mean()
    std_dev_test_ndcg = test_ndcgs.std()
    mean_test_hm = test_hms.mean()
    std_dev_test_hm = test_hms.std()

    # adjust best_validation_hit_rate for ItemPopularity model
    if isinstance(NET, ItemPopularity):
        best_validation_hit_rate = mean_test_hit_rate
        best_validation_ndcg = mean_test_ndcg
        best_validation_hm = mean_test_hm

    # store test hit-rate histogram
    WRITER.add_histogram('hit_rate/test/hist',
                         test_hit_rates.data,
                         global_step=best_validation_hit_rate)
    WRITER.add_histogram('ndcg/test/hist',
                         test_ndcgs.data,
                         global_step=best_validation_ndcg)
    WRITER.add_histogram('hm/test/hist',
                         test_hms.data,
                         global_step=best_validation_hm)

    # write scatter-plot dot (x=validation HR, y=mean test HR)
    WRITER.add_scalar('hit_rate/scatter', mean_test_hit_rate.item(),
                      global_step=best_validation_hit_rate)
    WRITER.add_scalar('ndcg/scatter', mean_test_ndcg.item(),
                      global_step=best_validation_ndcg)
    WRITER.add_scalar('hm/scatter', mean_test_hm.item(),
                      global_step=best_validation_hm)

    # print test results
    logger.info('\n---Test Results---')
    logger.info('HR@k:   \t%.2f%% (+/-%.4f%%)' %
                (mean_test_hit_rate, std_dev_test_hit_rate))
    logger.info('NDCG@k: \t%.2f%% (+/-%.4f%%)' %
                (mean_test_ndcg, std_dev_test_ndcg))
    logger.info('HM@k:   \t%.2f%% (+/-%.4f%%)' %
                (mean_test_hm, std_dev_test_hm))

    # EXECUTION TIME TRACKING ##################################################


    script_end_time = time.time()
    script_execution_time = script_end_time - script_start_time

    seconds = script_execution_time
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    exec_time_str = '{:02}:{:02}:{:02}'\
        .format(int(hours), int(minutes), int(seconds))

    logger.info('\n---Execution Time---')
    logger.info(exec_time_str+' (HH:MM:SS)')


    # QUINTUPLE SERVICE TEARDOWN ###############################################


    # random call that will never be true s.t. the quintuple service is not
    # removed by the compiler and stays alive over all batch loaders.
    # this avoids having multiple copies of the quintuples in memory
    # and also avoids loading and unloading them several times.
    if Q.get(0) is None:
        print('None!')


    # RETURN RESULTS ###########################################################

    if NET_NAME != 'ItemPopularity' and DO_VALIDATION:
        RESULTS['BEST_VALIDATION_EPOCH'] = best_validation_epoch
        RESULTS['BEST_VALIDATION_LOSS'] = best_validation_loss.item()
        RESULTS['BEST_VALIDATION_HIT_RATE'] = best_validation_hit_rate.item()
        RESULTS['BEST_VALIDATION_NDCG'] = best_validation_ndcg.item()
        RESULTS['BEST_VALIDATION_HM'] = best_validation_hm.item()

    RESULTS['TEST_HIT_RATES'] = test_hit_rates.cpu().numpy().tolist()
    RESULTS['MEAN_TEST_HIT_RATE'] = mean_test_hit_rate.item()
    RESULTS['STD_DEV_TEST_HIT_RATE'] = std_dev_test_hit_rate.item()
    RESULTS['TEST_NDCGS'] = test_ndcgs.cpu().numpy().tolist()
    RESULTS['MEAN_TEST_NDCG'] = mean_test_ndcg.item()
    RESULTS['STD_DEV_TEST_NDCG'] = std_dev_test_ndcg.item()
    RESULTS['TEST_HMS'] = test_hms.cpu().numpy().tolist()
    RESULTS['MEAN_TEST_HM'] = mean_test_hm.item()
    RESULTS['STD_DEV_TEST_HM'] = std_dev_test_hm.item()

    return RESULTS