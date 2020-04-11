import category_encoders as ce



def encoding(dict_encod):

    for name_encod in list(dict_encod.keys()):
        encoder =getattr(ce, name_encod)(cols=dict_encod[name_encod])


    return encoder
        





