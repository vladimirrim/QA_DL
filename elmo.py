from allennlp.modules.elmo import Elmo


def create_elmo(config):
    elmo = Elmo(config.ELMO_CONFIG, config.ELMO_WEIGHTS, 2, dropout=0)
    elmo = elmo.cuda()
    return elmo
