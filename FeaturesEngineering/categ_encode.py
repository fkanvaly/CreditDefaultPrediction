from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
import pandas as pd


class Encode:
    
    def __init__(self,data):
         self.data=data   
    
         self.dict_encoding={}
    

    def get(self):

        return self.data

    def LabelEncoding(self,args):
        self.dict_encoding["Label"]={}
        label=LabelEncoder()
        for col_name in args:
            if(self.data[col_name].dtype=='object'):
                #import ipdb; ipdb.set_trace()
                self.data[col_name]=label.fit_transform(self.data[col_name])
                
            else:
                print("{} is not an object".format(col_name))
        
        return self.data

    def OneHotEncoder(self,args):
        one=OneHotEncoder()
        for col_name in args:
            #import ipdb; ipdb.set_trace()
            if(self.data[col_name].dtype=='object'):
                    # import ipdb; ipdb.set_trace()

                    feat_one=pd.get_dummies(self.data[col_name])
                    
                    self.data=pd.concat([self.data,pd.DataFrame(feat_one)],axis=1)
            else:
                print("{} is not an object".format(col_name))
        self.data=self.data.drop(args,axis=1)
        return self.data

    def FeatureHasher(self,n_features,args):
        """[summary]
        Useful for very large categorical features
         
        Returns:
            [type] -- [description]
        """
        print("## Risk of mismatch due to NANs values !!!!")
        
        for col in args:

            try:
                self.data[col]=self.data[col].astype('str')  
                fh = FeatureHasher(n_features=n_features, input_type='string')
                feat_trans=fh.fit_transform(self.data[col])
                hashed_feat=feat_trans.toarray()
                self.data=pd.concat([ self.data , pd.DataFrame(hashed_feat,columns=[str(col)+"_{}".format(i) for i in range(n_features)]) ],axis=1) 
                
                print(col)
                
            except :
                import ipdb; ipdb.set_trace()
        self.data=self.data.drop(args,axis=1)

        
        return self.data

    def ordinary_encoding(self,args,dict_args):
        """[Like Labelencoding but considering variables as ordinal variables]
        
        Arguments:
            args =[]-- List of columns
            dict_args {col_1:{value_1:1 ,value_2 : 2 ,..},col_1:{value_1:1 ,value_2 : 2 ,..} ,..} -- [Dictionnary of dictionnaries of variables values to be mapped]
        
        Returns:
            [type] -- [replace each variables values by his mapping value]
        """

        for col in args :
            self.data[col]=self.data[col].astype("str")
            # import ipdb ; ipdb.set_trace()
            self.data[col]=self.data[col].map(dict_args[col])
        


        return self.data



    def mean_encoding(self,args):
        "Link categorial variable to target"
        print("Need to save values for predicting")
        mean_dict={}
        for col in args :
            mean_encode=self.data.groupby(col)["Y"].mean()
            mean_dict[col]=mean_encode
            print("Need to save values for predicting")
            self.data[col]=self.data[col].map(mean_encode)
            # mean_dict[col]=mean_encode
        

        return self.data       
    def woe_encoding(self,args):
        self.dict_encoding["WOE"]={}
        for col in args:
            woe_df=self.data.groupby(col)["Y"].mean()
            woe_df=pd.Dataframe(woe_df)
            woe_df=woe_df.rename(colums={"Y":"Good"})

            woe_df["Bad"]=1-woe_df.Good
            woe_df["Bad"]=np.where(woe_df["Bad"]==0,0.00001,woe_df["Bad"])
    
            self.data[col]=np.log(woe_df.Good/woe_df.Bad)


        return self.data

    def probaratio_enco(self,args):

        for col in args:
            woe_df=self.data.groupby(col)["Y"].mean()
            woe_df=pd.Dataframe(woe_df)
            woe_df=woe_df.rename(colums={"Y":"Good"})

            woe_df["Bad"]=1-woe_df.Good
            woe_df["Bad"]=np.where(woe_df["Bad"]==0,0.00001,woe_df["Bad"])
            self.data[col]=woe_df.Good/woe_df.Bad

        return self.data


    def

    

       


        
        