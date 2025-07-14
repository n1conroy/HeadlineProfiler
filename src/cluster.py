import utils
import model

class Cluster:
    def __init__(self, document, id=None):
        self.clustid = id
        self.ids = set()
        self.num_docs = 0
        self.reprs = {}
        self.sum_timestamp = 0
        self.sumsq_timestamp = 0
        self.newest_timestamp = utils.datetime.datetime.strptime(
            "1000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.oldest_timestamp = utils.datetime.datetime.strptime(
            "3000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

        # A cluster must be created with at least one document included
        self.add_document(document)  
        
    def get_relevance_stamp(self):
        z_score = 1
        mean = self.sum_timestamp / self.num_docs
        try:
          std_dev = utils.math.sqrt((self.sumsq_timestamp / self.num_docs) - (mean*mean))
        except:
          std_dev = 0.0
        return mean + ((z_score * std_dev) * 3600.0) # its in secods since epoch

    def add_document(self, document):
        self.ids.add(document.id)
        self.newest_timestamp = max(self.newest_timestamp, document.timestamp)
        self.oldest_timestamp = min(self.oldest_timestamp, document.timestamp)
        ts_hours =  (document.timestamp.timestamp() / 3600.0)
        self.sum_timestamp += ts_hours
        self.sumsq_timestamp += ts_hours * ts_hours
        self.__add_reps(document.reprs)
   
    def __add_reps(self, reprs0):

        if self.reprs: 
            for n in model.model_reps:
                self.reprs[n] = utils.np.nanmean(utils.np.array([self.reprs[n], reprs0[n]]), axis=0)
        else:
            self.reprs = reprs0        
        self.num_docs += 1


class Aggregator:
    def __init__(self, thr, SVMmodel: model.Model, NNmodel: model.Model):
        self.clusterpool = []
        self.SVMmodel = SVMmodel
        self.NNmodel = NNmodel.load_NN()
        self.thr = thr
        self.cluster_num = 0

    def load_clusterpool(self, clusterpool_loc):
        self.clusterpool= utils.pd.read_pickle(clusterpool_loc)
        self.cluster_num = len(self.clusterpool)

    def sort_clusterpool(self):
        self.clusterpool.sort(key=lambda cluster: cluster.clustid)

    def PutDocument(self, document, mode):
        ''' 
        Decides on which cluster to add the new document to based on model score
        '''
        t0 = utils.time.time()
        best_i = -1
        best_s = 0.0
        new_cluster =0 
        i = -1
        correct = False
        features = []

        reps = {}
        print ("Document: (", document.id, ') ', document.title)
        print ('ACTUAL CLUSTER: ', document.cluster)
        print ('ACTUAL TIMESTAMP: ', document.timestamp)

        '''
            INSTANTIATE A NEW MODEL
        '''
        for cluster in self.clusterpool:
            features = []
            i += 1
            reps = utils.sim_reps_dc(document, cluster)
            score = utils.c_score(reps, self.SVMmodel)
            
            if mode == 'eval':
                if score < self.thr:
                    features.append(0)
                else:
                    features.append(score)
                features.extend(reps.values())
                new_cluster = self.NNmodel.predict(utils.np.array( [features,] ) )

            '''
            Make a prediction for a new cluster candidate or not
            '''
            if score > best_s and (new_cluster < 0.9):
            #if score > best_s and (score > self.thr):
                best_s = score
                best_i = i
                best_rep = reps
                label = 0
            
        '''
        Just label cluster numbers for new clusters as their order of creation
        '''
        
        if (best_i == -1):
            print ('best score', best_s)
            self.cluster_num = document.cluster
            print ('No close matches, creating new cluster, CLUSTER ', self.cluster_num, ' in the pool.')

            self.clusterpool.append(Cluster(document, self.cluster_num))
            best_i = len(self.clusterpool) - 1
            best_rep = reps
            label = 1
        else:
            print ("\nThe best cscore from clusterpool: ", best_s)
            print ('adding to cluster:', self.clusterpool[best_i].clustid, "with timestamp ", self.clusterpool[best_i].newest_timestamp)            
            self.clusterpool[best_i].add_document(document)

        t1 = utils.time.time()
        self.sort_clusterpool()
        print ("Clusterization time: ", t1-t0, '\n\n')

        if (document.cluster == self.cluster_num) or (document.cluster == self.clusterpool[best_i].clustid):
            correct = True
        
        print ('The prediction is ', correct)

        if mode != 'eval':
            features = []
            features.append(best_s)
            features.extend(best_rep.values())

        clftime = t1-t0
        return ([label,features], clftime, correct)




