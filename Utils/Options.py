class Param:
    def __init__(self):
        self.root = ''
        self.db_path = f'{self.root}/'
        self.save_path = f'{self.root}/BYOL/Backup/try_1'

        self.ckp_path = f'{self.save_path}/ckp'
        self.log_path = f'{self.save_path}/log'

        self.device = 'cuda'
        self.batchsz = 16
        self.lr = 1e-4
        self.full_epoch = 100

        self.do_knn = True