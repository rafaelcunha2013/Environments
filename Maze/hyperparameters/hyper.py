def hyperparameter():
    GAMMA=0.99
    BATCH_SIZE=64
    BUFFER_SIZE=100*1000
    MIN_REPLAY_SIZE=1000
    EPSILON_START=1.0
    EPSILON_END=0.01
    EPSILON_DECAY=100*1000
    TARGET_UPDATE_FREQ=1000*20
    LEARNING_RATE = 1e-4
    HIDDEN_DIM = 256

    parameters = {}
    parameters['GAMMA'] = GAMMA
    parameters['BATCH_SIZE'] = BATCH_SIZE
    parameters['BUFFER_SIZE'] = BUFFER_SIZE
    parameters['MIN_REPLAY_SIZE'] = MIN_REPLAY_SIZE
    parameters['EPSILON_START'] = EPSILON_START
    parameters['EPSILON_END'] = EPSILON_END
    parameters['EPSILON_DECAY'] = EPSILON_DECAY
    parameters['TARGET_UPDATE_FREQ'] = TARGET_UPDATE_FREQ
    parameters['LEARNING_RATE'] = LEARNING_RATE
    parameters['HIDDEN_DIM'] = HIDDEN_DIM

    return parameters