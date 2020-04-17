import category_encoders as ce

import sys
sys.path.append("../../CreditDefaultPrediction")

from FeaturesEngineering.preprocessing import *
from utils.sampling import *

def encoding(dict_encod):
    """Categorical variable encoding
    
    Arguments:
        dict_encod {dict} -- organize like {"encoder_name": [list of col], ...}
    
    Returns:
        transformer -- transformer to apply to apply to data to encode it
    """
    for name_encod in list(dict_encod.keys()):
        encoder = getattr(ce, name_encod)(cols=dict_encod[name_encod])


    return encoder
        

if __name__ == "__main__":

    df = prepocess_data("../data/raw/CreditTraining.csv")
    df = random_under_sampling(df)

    y = df.Y
    X = df.drop("Y", axis=1)

    cat = [col for col in df if df[col].dtype.name == 'category']
    dict_encod = {"WOEEncoder": cat}

    encoder = encoding(dict_encod)
    X_enc = encoder.fit_transform(X,y)
    import ipdb; ipdb.set_trace()    