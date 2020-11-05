import pandas as pd
import numpy as np

@pd.api.extensions.register_dataframe_accessor("peer")
class PeerCompare:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # validate the pandas dataframe 
        pass


    def naive_peer_map(self,name_of_peer_map:str,peer_attributes:list)->None:
        # adds naive peer groups to the dataframe based on the peer attributes if all 
        # the peer attributes are columns of the dataframe
        check_attr = [attr not in self._obj.columns for attr in peer_attributes]
        check_attr = sum(check_attr)
        if check_attr>0:
            raise AttributeError("All peer attributes must be column names of the dataframe.")
        self._obj[peer_attributes] = self._obj[peer_attributes].astype(str)
        self._obj[name_of_peer_map] = self._obj[peer_attributes].agg('_'.join, axis=1)

    def exhaustive_peer_map(self)->None:
        # adds a column to the dataframe called 'exhaustive_peer_map'
        print("A column with name 'exhaustive_peer_map' has been added to your dataframe.")
        self._obj['exhaustive_peer_map'] = np.arange(0,self._obj.shape[0])
    
    def lazy_peer_map(self)->None:
        # adds a column to the d
        print("A column with name 'lazy_peer_map' has been added to your dataframe.")
        self._obj['lazy_peer_map'] = 'lazy_peer_group'


    def peer_map_quantiles(self,peer_map_name:str,quantiles:list,target:str)->pd.DataFrame:
        # computes peer map quantiles if the peer map already exist
        quantiles = np.unique(np.append(quantiles,[0,1]))
        quantiles = np.sort(quantiles)

        check_quantile = (np.array([kk>1 for kk in np.array(quantiles)]) & np.array([kk<0 for kk in np.array(quantiles)]))
        if sum(check_quantile)>1: 
            raise AssertionError("Quantiles must be in [0,1]")

        check_attr = peer_map_name not in  self._obj.columns
        if check_attr:
            raise AttributeError("Peer map does not exist in the dataset yet, please create one.")
       
        if target not in self._obj:
            raise AttributeError("Target must be a valid column name.")
        dt_ = []
        dt = pd.DataFrame()
    
        if peer_map_name == 'exhaustive_peer_map':
            for q in quantiles:
                dt['Q_'+str(q)] = self._obj[target]
        else:
            for q in quantiles:
                dt = (self._obj.groupby(peer_map_name)
                        .agg(Q=(target,lambda x:np.quantile(x,q))))
                dt = dt.rename({'Q':'Q_'+str(q)},axis='columns')
                dt_.append(dt)
            return(pd.concat(dt_,axis=1))
  
  
    def quantile_map(self,peer_map_name:str,quantiles:list,target:str)->None:
        # quantile_map for the exhaustive case are the activity scores
     
        quantiles = np.unique(np.append(quantiles,[0,1]))
        quantiles = np.sort(quantiles)
        
        name = str(peer_map_name)+'_Q_value'
        if peer_map_name == 'exhaustive_peer_map':
            self._obj[name] = self._obj[target]
        else:
            self._obj[name] = None
            peer_map_quantiles = self.peer_map_quantiles(peer_map_name,quantiles,target)
            names = peer_map_quantiles.columns

            for i in range(len(names)):
                self._obj[names[i]] = self._obj[peer_map_name].replace(list(peer_map_quantiles.index),
                                                                       peer_map_quantiles[names[i]].values)

            bool_ = (self._obj[target] == self._obj[names[0]])        
            self._obj[name][bool_] = ((self._obj[names[0]]+self._obj[names[1]])/2)[bool_]

            for i in np.arange(1,len(names)):
                condtion_1 = self._obj[names[i]] == self._obj[names[i-1]]
                condtion_2 = ((self._obj[target] <= self._obj[names[i]]) & (self._obj[target] > self._obj[names[i-1]]))
                bool_ = condtion_1 | condtion_2
                self._obj[name][bool_] = ((self._obj[names[i-1]]+self._obj[names[i]])/2)[bool_]
            self._obj.drop(names,inplace=True,axis=1)

        def error(self,peer_map_names:list,target:str)->None:
            # TODO
            # compute error for each quantile mapped value aka prediction 
            pass

        def inverse_var()->pd.DataFrame:
            #TODO
            # computes inverse variance  for each peer group
            pass

        def accuracy(self,peer_map_names:list,quantiles:list,target:str)->pd.DataFrame:
            # TODO
            # compute accuary according description in overleaf
            pass


    

  
  


# test
dataset = pd.DataFrame({'A':np.random.binomial(3,0.03,N),
                        'B':np.random.binomial(4,1,N),
                        'C':np.random.normal(0,3,N)})


dataset.peer.exhaustive_peer_map()
dataset.peer.lazy_peer_map()
dataset.peer.naive_peer_map('naive_map',['A','B'])


dataset.peer.peer_map_quantiles('naive_map',[0.1,0.3,0.4],'C')
dataset.peer.quantile_map(peer_map_name='naive_map',quantiles=[0.1,0.3,0.2],target='C')

dataset